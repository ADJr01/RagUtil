import os
import json
import hashlib
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader
)
from langchain_core.documents import Document
from file_parsers.helper.parsing_helper import (
    clean_pdf_text as cleaner,
    is_pdf_file
)


class ParserConfigError(Exception):
    """Raised when parser configuration is invalid."""
    pass


class Parser:
    """
    Production-ready document parser with intelligent chunking and ChromaDB integration.

    Supports: .txt, .pdf, .xlsx, .csv, .doc, .docx files
    Features:
    - Automatic file tracking via record files
    - Intelligent synchronization (only processes changed files)
    - Cross-platform compatibility (Windows/Linux)
    - Configurable chunking parameters
    - Robust error handling

    Example:
        config = {
            'storage_dir': './documents',
            'persist_dir': './chroma_db',
            'collection_name': 'my_docs',
            'embedding': OpenAIEmbeddings(),
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'verbose': True
        }
        parser = Parser(config)
        vector_store = parser.load()
    """

    SUPPORTED_TYPES = {'.txt', '.pdf', '.xlsx', '.csv', '.doc', '.docx'}
    REQUIRED_CONFIG_KEYS = {'storage_dir', 'persist_dir', 'collection_name', 'embedding'}

    def __init__(self, config: Dict):
        """
        Initialize Parser with configuration dictionary.

        Args:
            config: Configuration dictionary with the following keys:
                - storage_dir (str): Directory containing files to parse [REQUIRED]
                - persist_dir (str): Directory for ChromaDB persistence [REQUIRED]
                - collection_name (str): Name of ChromaDB collection [REQUIRED]
                - embedding: Embedding function for ChromaDB [REQUIRED]
                - chunk_size (int): Size of text chunks (default: 1000)
                - chunk_overlap (int): Overlap between chunks (default: 200)
                - is_separator_regex (bool): Use regex separators (default: False)
                - verbose (bool): Enable logging output (default: True)

        Raises:
            ParserConfigError: If required configuration is missing or invalid
        """
        self._validate_config(config)
        self._initialize_config(config)
        self._initialize_paths()
        self._initialize_text_splitter()
        self._initialize_state()
        self._initialize_collection()

    def _validate_config(self, config: Dict):
        """Validate configuration dictionary."""
        if not isinstance(config, dict):
            raise ParserConfigError("Config must be a dictionary")

        missing_keys = self.REQUIRED_CONFIG_KEYS - set(config.keys())
        if missing_keys:
            raise ParserConfigError(f"Missing required config keys: {missing_keys}")

        # Validate types
        if not isinstance(config.get('storage_dir'), str):
            raise ParserConfigError("'storage_dir' must be a string")
        if not isinstance(config.get('persist_dir'), str):
            raise ParserConfigError("'persist_dir' must be a string")
        if not isinstance(config.get('collection_name'), str):
            raise ParserConfigError("'collection_name' must be a string")
        if not config.get('collection_name').strip():
            raise ParserConfigError("'collection_name' cannot be empty")

    def _initialize_config(self, config: Dict):
        """Initialize configuration with defaults."""
        self.storage_dir = Path(config['storage_dir']).resolve()
        self.persist_dir = Path(config['persist_dir']).resolve()
        self.collection_name = config['collection_name'].strip()
        self.embedding = config['embedding']

        # Optional parameters with defaults
        self.chunk_size = config.get('chunk_size', 1000)
        self.chunk_overlap = config.get('chunk_overlap', 200)
        self.is_separator_regex = config.get('is_separator_regex', False)
        self.verbose = config.get('verbose', True)

        # Validate numeric parameters
        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ParserConfigError("'chunk_size' must be a positive integer")
        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ParserConfigError("'chunk_overlap' must be a non-negative integer")
        if self.chunk_overlap >= self.chunk_size:
            raise ParserConfigError("'chunk_overlap' must be less than 'chunk_size'")

    def _initialize_paths(self):
        """Create necessary directories."""
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ParserConfigError(f"Failed to create directories: {e}")

        self.record_file = self._get_record_file_path()

    def _initialize_text_splitter(self):
        """Initialize text splitter with configured parameters."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=self.is_separator_regex
        )

    def _initialize_state(self):
        """Initialize internal state."""
        self._vector_store: Optional[Chroma] = None
        self._file_records: Dict[str, Dict] = {}

    def _log(self, message: str, level: str = 'info'):
        """
        Log message if verbose mode is enabled.

        Args:
            message: Message to log
            level: Log level ('info', 'warning', 'error')
        """
        if not self.verbose and level == 'info':
            return

        if level == 'warning':
            warnings.warn(message)
        elif level == 'error':
            print(f"ERROR: {message}")
        else:
            print(message)

    def _get_record_file_path(self) -> Path:
        """
        Generate unique record file path based on collection and storage directory.

        Returns:
            Path to record file
        """
        unique_id = hashlib.md5(
            f"{self.collection_name}_{self.storage_dir}".encode()
        ).hexdigest()[:16]

        record_filename = f"record_{self.collection_name}_{unique_id}.json"
        return self.persist_dir / record_filename

    def _load_record_file(self) -> Dict[str, Dict]:
        """
        Load record file containing file tracking information.

        Returns:
            Dictionary with file paths as keys and metadata as values
        """
        if not self.record_file.exists():
            return {}

        try:
            with open(self.record_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
            self._log(f"Loaded record file with {len(records)} entries")
            return records
        except json.JSONDecodeError as e:
            self._log(f"Record file corrupted: {e}", 'error')
            return {}
        except Exception as e:
            self._log(f"Error loading record file: {e}", 'error')
            return {}

    def _save_record_file(self):
        """Save current file records to disk."""
        try:
            # Atomic write: write to temp file then rename
            temp_file = self.record_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self._file_records, f, indent=2, ensure_ascii=False)

            # Atomic rename (works on both Windows and Linux)
            temp_file.replace(self.record_file)
            self._log(f"Record file saved with {len(self._file_records)} entries")
        except Exception as e:
            self._log(f"Error saving record file: {e}", 'error')
            if temp_file.exists():
                temp_file.unlink()

    def _get_file_hash(self, file_path: Path) -> str:
        """
        Calculate MD5 hash of a file.

        Args:
            file_path: Path to file

        Returns:
            MD5 hash as hexadecimal string
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self._log(f"Error hashing file {file_path.name}: {e}", 'error')
            return ""

    def _collection_exists(self) -> bool:
        """
        Check if ChromaDB collection exists.

        Returns:
            True if collection exists, False otherwise
        """
        try:
            test_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=str(self.persist_dir)
            )
            return test_store._collection.count() >= 0
        except Exception:
            return False

    def _initialize_collection(self):
        """
        Initialize collection on startup with proper validation.

        Handles four scenarios:
        1. No collection, no record: Fresh start
        2. Collection exists, no record: Delete collection (corrupted state)
        3. Collection and record exist: Load and validate
        4. No collection, record exists: Rebuild on load()
        """
        collection_exists = self._collection_exists()
        record_exists = self.record_file.exists()

        self._log(f"Initializing collection '{self.collection_name}'")
        self._log(f"  Collection exists: {collection_exists}")
        self._log(f"  Record file exists: {record_exists}")

        if collection_exists and not record_exists:
            # Corrupted state: delete collection
            self._log("Collection exists without record file. Deleting for clean state.", 'warning')
            self._delete_collection_internal()
            self._file_records = {}
            self._vector_store = None

        elif collection_exists and record_exists:
            # Normal state: load both
            self._file_records = self._load_record_file()
            try:
                self._vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embedding,
                    persist_directory=str(self.persist_dir)
                )
                doc_count = self._vector_store._collection.count()
                self._log(f"Loaded collection with {doc_count} documents")
            except Exception as e:
                self._log(f"Error loading collection: {e}", 'error')
                self._vector_store = None

        elif not collection_exists and record_exists:
            # Record exists but no collection: will rebuild
            self._log("Record file exists but collection missing. Will rebuild on load()")
            self._file_records = self._load_record_file()
            self._vector_store = None

        else:
            # Fresh start
            self._log("Fresh start: No existing collection or record")
            self._file_records = {}
            self._vector_store = None

    def _delete_collection_internal(self):
        """Internal method to delete collection."""
        try:
            temp_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=str(self.persist_dir)
            )
            temp_store.delete_collection()
            self._log(f"Collection '{self.collection_name}' deleted")
        except Exception as e:
            self._log(f"Error deleting collection: {e}", 'error')

    def clean_pdf_text(self, text: str) -> str:
        """
        Clean PDF/Doc text while preserving code syntax.

        Args:
            text: Raw text from PDF/Doc

        Returns:
            Cleaned text
        """
        return cleaner(text)

    def _load_file(self, file_path: Path) -> List[Document]:
        """
        Load a file based on its extension.

        Args:
            file_path: Path to file

        Returns:
            List of Document objects
        """
        ext = file_path.suffix.lower()
        file_path_str = str(file_path.resolve())

        try:
            if ext == '.txt':
                loader = TextLoader(file_path_str, encoding='utf-8')
                documents = loader.load()

            elif ext == '.pdf':
                loader = PyPDFLoader(file_path_str)
                documents = loader.load()
                for doc in documents:
                    doc.page_content = self.clean_pdf_text(doc.page_content)

            elif ext == '.xlsx':
                loader = UnstructuredExcelLoader(file_path_str, mode="elements")
                documents = loader.load()

            elif ext == '.csv':
                loader = CSVLoader(file_path_str, encoding='utf-8')
                documents = loader.load()

            elif ext == '.docx':
                loader = Docx2txtLoader(file_path_str)
                documents = loader.load()
                for doc in documents:
                    doc.page_content = self.clean_pdf_text(doc.page_content)

            elif ext == '.doc':
                loader = UnstructuredWordDocumentLoader(file_path_str)
                documents = loader.load()
                for doc in documents:
                    doc.page_content = self.clean_pdf_text(doc.page_content)
            else:
                return []

            return documents

        except Exception as e:
            self._log(f"Error loading {file_path.name}: {e}", 'error')
            return []

    def _get_all_files(self) -> List[Path]:
        """
        Get all supported files from storage directory.

        Returns:
            Sorted list of file paths
        """
        files = []
        for ext in self.SUPPORTED_TYPES:
            files.extend(self.storage_dir.glob(f"*{ext}"))
        return sorted(files)

    def _create_smart_metadata(
            self,
            doc: Document,
            file_path: Path,
            chunk_idx: int,
            total_chunks: int,
            file_hash: str
    ) -> Dict:
        """
        Create metadata for each chunk.

        Args:
            doc: Document object
            file_path: Path to source file
            chunk_idx: Chunk index
            total_chunks: Total chunks for document
            file_hash: File hash

        Returns:
            Metadata dictionary
        """
        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'file_type': file_path.suffix.lower(),
            'file_size': file_path.stat().st_size,
            'file_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'chunk_index': chunk_idx,
            'total_chunks': total_chunks,
            'chunk_position': f"{chunk_idx + 1}/{total_chunks}",
            'is_first_chunk': chunk_idx == 0,
            'is_last_chunk': chunk_idx == total_chunks - 1,
            'char_count': len(doc.page_content),
            'word_count': len(doc.page_content.split()),
            'processed_at': datetime.now().isoformat(),
            'file_hash': file_hash,
        }

        # Merge with existing metadata
        if hasattr(doc, 'metadata') and doc.metadata:
            metadata.update({k: v for k, v in doc.metadata.items() if k not in metadata})

        return metadata

    def _process_documents(
            self,
            documents: List[Document],
            file_path: Path,
            file_hash: str
    ) -> List[Document]:
        """
        Split documents into chunks with metadata.

        Args:
            documents: List of documents
            file_path: Path to source file
            file_hash: File hash

        Returns:
            List of chunked documents
        """
        full_text = "\n\n".join([doc.page_content for doc in documents])
        chunks = self.text_splitter.create_documents([full_text])
        total_chunks = len(chunks)

        processed_docs = []
        for idx, chunk in enumerate(chunks):
            chunk.metadata = self._create_smart_metadata(
                chunk, file_path, idx, total_chunks, file_hash
            )
            processed_docs.append(chunk)

        return processed_docs

    def _sync_with_storage(self) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Compare record file with storage directory.

        Returns:
            Tuple of (new_files, modified_files, deleted_files)
        """
        current_files = {str(f): f for f in self._get_all_files()}
        recorded_files = set(self._file_records.keys())

        new_files = set(current_files.keys()) - recorded_files
        deleted_files = recorded_files - set(current_files.keys())

        modified_files = set()
        for file_path_str in recorded_files & set(current_files.keys()):
            current_hash = self._get_file_hash(Path(file_path_str))
            if current_hash and current_hash != self._file_records[file_path_str].get('hash', ''):
                modified_files.add(file_path_str)

        return new_files, modified_files, deleted_files

    def _ensure_vector_store(self):
        """Ensure vector store is initialized."""
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=str(self.persist_dir)
            )

    def load(self) -> Chroma:
        """
        Load and process files with intelligent synchronization.
        Only processes new or modified files.

        Returns:
            Chroma vector store instance
        """
        self._log(f"\nLoading from {self.storage_dir}")

        new_files, modified_files, deleted_files = self._sync_with_storage()

        self._log(f"  New: {len(new_files)} | Modified: {len(modified_files)} | Deleted: {len(deleted_files)}")
        self._log(f"  Up-to-date: {len(self._file_records) - len(modified_files) - len(deleted_files)}")

        if not (new_files or modified_files or deleted_files):
            self._log("All files up-to-date. Skipping processing.")
            self._ensure_vector_store()
            return self._vector_store

        self._ensure_vector_store()

        # Process deletions
        for file_path_str in deleted_files:
            self._log(f"  Removing: {Path(file_path_str).name}")
            try:
                self._vector_store.delete(where={"source": file_path_str})
                del self._file_records[file_path_str]
            except Exception as e:
                self._log(f"    Error: {e}", 'error')

        # Process modifications
        for file_path_str in modified_files:
            file_path = Path(file_path_str)
            self._log(f"  Updating: {file_path.name}")

            try:
                self._vector_store.delete(where={"source": file_path_str})

                documents = self._load_file(file_path)
                if documents:
                    file_hash = self._get_file_hash(file_path)
                    if file_hash:
                        chunks = self._process_documents(documents, file_path, file_hash)
                        self._vector_store.add_documents(chunks)

                        self._file_records[file_path_str] = {
                            'hash': file_hash,
                            'filename': file_path.name,
                            'processed_at': datetime.now().isoformat(),
                            'chunk_count': len(chunks)
                        }
            except Exception as e:
                self._log(f"    Error: {e}", 'error')

        # Process new files
        for file_path_str in new_files:
            file_path = Path(file_path_str)
            self._log(f"  Adding: {file_path.name}")

            try:
                documents = self._load_file(file_path)
                if documents:
                    file_hash = self._get_file_hash(file_path)
                    if file_hash:
                        chunks = self._process_documents(documents, file_path, file_hash)
                        self._vector_store.add_documents(chunks)

                        self._file_records[file_path_str] = {
                            'hash': file_hash,
                            'filename': file_path.name,
                            'processed_at': datetime.now().isoformat(),
                            'chunk_count': len(chunks)
                        }
            except Exception as e:
                self._log(f"    Error: {e}", 'error')

        self._save_record_file()

        self._log(f"\nLoad complete. Tracked files: {len(self._file_records)}")
        self._log(f"Collection documents: {self._vector_store._collection.count()}")

        return self._vector_store

    def reload(self) -> Chroma:
        """
        Reload and synchronize with storage directory.

        Returns:
            Updated Chroma vector store
        """
        self._log("\n" + "=" * 60)
        self._log("RELOAD: Checking for changes")
        self._log("=" * 60)
        return self.load()

    def drop(self) -> bool:
        """
        Delete collection and record file.

        Returns:
            True if successful, False otherwise
        """
        try:
            self._delete_collection_internal()

            if self.record_file.exists():
                self.record_file.unlink()
                self._log(f"Record file deleted: {self.record_file.name}")

            self._vector_store = None
            self._file_records.clear()

            self._log(f"Collection '{self.collection_name}' and records deleted")
            return True
        except Exception as e:
            self._log(f"Error during drop: {e}", 'error')
            return False

    def get_vector_store(self) -> Chroma:
        """
        Get ChromaDB vector store instance.

        Returns:
            Chroma vector store
        """
        self._ensure_vector_store()
        return self._vector_store

    def get_file_records(self) -> Dict[str, Dict]:
        """
        Get current file records.

        Returns:
            Copy of file records dictionary
        """
        return self._file_records.copy()

    def get_stats(self) -> Dict:
        """
        Get parser statistics.

        Returns:
            Statistics dictionary
        """
        storage_files = self._get_all_files()

        stats = {
            'collection_name': self.collection_name,
            'storage_dir': str(self.storage_dir),
            'persist_dir': str(self.persist_dir),
            'record_file': str(self.record_file),
            'record_file_exists': self.record_file.exists(),
            'files_in_storage': len(storage_files),
            'files_in_records': len(self._file_records),
            'collection_exists': self._collection_exists(),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
        }

        if self._vector_store:
            try:
                stats['collection_document_count'] = self._vector_store._collection.count()
            except Exception:
                stats['collection_document_count'] = 0
        else:
            stats['collection_document_count'] = 0

        return stats



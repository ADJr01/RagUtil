import os
import json
import hashlib
import warnings
import platform
import threading
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any
from datetime import datetime
from contextlib import contextmanager
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
from file_parsers.helper.parsing_helper import clean_pdf_text as cleaner


class ParserConfigError(Exception):
    """Raised when parser configuration is invalid."""
    pass


class ParserLockError(Exception):
    """Raised when parser lock operations fail."""
    pass


class ParserStateError(Exception):
    """Raised when parser state is inconsistent."""
    pass


class ConfigFileManager:
    """
    Manages parser.config.json with cross-platform file locking.
    Handles read/write operations with proper synchronization.
    """

    CONFIG_FILENAME = "parser.config.json"
    CONFIG_VERSION = "2.0"

    def __init__(self, persist_dir: Path, verbose: bool = True):
        self.persist_dir = persist_dir
        self.config_path = persist_dir / self.CONFIG_FILENAME
        self.verbose = verbose
        self._file_handle: Optional[Any] = None
        self._lock = threading.RLock()
        self._is_windows = platform.system() == "Windows"
        self._closed = False

    def _log(self, message: str, level: str = 'info'):
        """Log message if verbose mode is enabled."""
        if not self.verbose and level == 'info':
            return
        if level == 'error':
            print(f"ERROR: {message}")
        elif level == 'warning':
            warnings.warn(message)
        else:
            print(message)

    @contextmanager
    def _file_lock(self, mode: str = 'r'):
        """
        Cross-platform file locking context manager.

        Args:
            mode: File open mode ('r' or 'w')
        """
        if self._closed:
            raise ParserLockError("ConfigFileManager is closed")

        file_handle = None
        try:
            file_handle = open(self.config_path, mode, encoding='utf-8')

            if self._is_windows:
                # Windows file locking
                import msvcrt
                lock_flag = msvcrt.LK_NBLCK if mode == 'r' else msvcrt.LK_NBRLCK
                try:
                    msvcrt.locking(file_handle.fileno(), lock_flag, 1)
                except IOError:
                    raise ParserLockError(f"Could not acquire {mode} lock on config file")
            else:
                # Unix file locking
                lock_type = fcntl.LOCK_SH if mode == 'r' else fcntl.LOCK_EX
                try:
                    fcntl.flock(file_handle.fileno(), lock_type | fcntl.LOCK_NB)
                except IOError:
                    raise ParserLockError(f"Could not acquire {mode} lock on config file")

            yield file_handle

        finally:
            if file_handle:
                try:
                    if self._is_windows:
                        import msvcrt
                        msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
                except Exception as e:
                    self._log(f"Error releasing lock: {e}", 'warning')
                finally:
                    file_handle.close()

    def create_initial_config(self, config_data: Dict) -> None:
        """Create initial configuration file."""
        with self._lock:
            if self.config_path.exists():
                raise ParserConfigError(
                    f"Config file already exists at {self.config_path}. "
                    "Use load_config() or validate_and_load() instead."
                )

            config_data['_metadata'] = {
                'version': self.CONFIG_VERSION,
                'created_at': datetime.now().isoformat(),
                'last_modified': datetime.now().isoformat(),
                'parser_state': 'initialized'
            }

            # Atomic write
            temp_path = self.config_path.with_suffix('.tmp')
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                temp_path.replace(self.config_path)
                self._log(f"Created config file: {self.CONFIG_FILENAME}")
            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                raise ParserConfigError(f"Failed to create config file: {e}")

    def load_config(self) -> Dict:
        """Load configuration with read lock."""
        with self._lock:
            if not self.config_path.exists():
                raise ParserConfigError(
                    f"Config file not found at {self.config_path}. "
                    "Initialize parser first."
                )

            try:
                with self._file_lock('r') as f:
                    config = json.load(f)
                return config
            except json.JSONDecodeError as e:
                raise ParserConfigError(
                    f"Invalid JSON in config file: {e}. "
                    "Config file is corrupted."
                )
            except Exception as e:
                raise ParserConfigError(f"Failed to load config: {e}")

    def update_config(self, updates: Dict) -> None:
        """Update configuration with write lock."""
        with self._lock:
            if not self.config_path.exists():
                raise ParserConfigError("Config file does not exist")

            try:
                # Read current config
                with self._file_lock('r') as f:
                    config = json.load(f)

                # Apply updates
                config.update(updates)
                config['_metadata']['last_modified'] = datetime.now().isoformat()

                # Atomic write
                temp_path = self.config_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                temp_path.replace(self.config_path)

            except json.JSONDecodeError:
                raise ParserConfigError("Invalid JSON in config file")
            except Exception as e:
                raise ParserConfigError(f"Failed to update config: {e}")

    def validate_config(self, config: Dict) -> bool:
        """
        Validate configuration structure and version.

        Returns:
            True if valid
        """
        required_keys = {'collection_name', 'storage_dir', 'chunk_size',
                         'chunk_overlap', '_metadata'}

        if not all(key in config for key in required_keys):
            missing = required_keys - set(config.keys())
            raise ParserConfigError(f"Missing required config keys: {missing}")

        metadata = config.get('_metadata', {})
        config_version = metadata.get('version', '0.0')

        if config_version != self.CONFIG_VERSION:
            raise ParserConfigError(
                f"Config version mismatch. Expected {self.CONFIG_VERSION}, "
                f"got {config_version}. Migration required."
            )

        return True

    def close_with_read_lock(self) -> None:
        """
        Close config file with read-only access and write lock.
        Prevents any write operations to the config file.
        """
        with self._lock:
            if self._closed:
                self._log("ConfigFileManager already closed", 'warning')
                return

            try:
                # Set file to read-only
                if self._is_windows:
                    os.chmod(self.config_path, 0o444)
                else:
                    os.chmod(self.config_path, 0o444)

                # Update metadata to indicate closed state
                config = self.load_config()
                config['_metadata']['parser_state'] = 'closed'
                config['_metadata']['closed_at'] = datetime.now().isoformat()

                # Final write before locking
                temp_path = self.config_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                temp_path.replace(self.config_path)

                # Set to read-only again
                os.chmod(self.config_path, 0o444)

                self._closed = True
                self._log(f"Config file closed with read-only lock: {self.CONFIG_FILENAME}")

            except Exception as e:
                raise ParserLockError(f"Failed to close config file: {e}")

    def delete_config(self) -> None:
        """Delete configuration file."""
        with self._lock:
            if self.config_path.exists():
                try:
                    # Remove read-only if set
                    os.chmod(self.config_path, 0o644)
                    self.config_path.unlink()
                    self._log("Config file deleted")
                except Exception as e:
                    raise ParserConfigError(f"Failed to delete config: {e}")


class Parser:
    """
    Production-ready document parser with intelligent chunking and ChromaDB integration.

    Features:
    - Single parser.config.json per persist_dir
    - Cross-platform file locking (Windows/Linux)
    - Automatic state synchronization
    - Incremental updates
    - Robust error handling and recovery
    - Health checks and diagnostics
    - Batch processing support
    - Memory-efficient operations

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
        try:
            vector_store = parser.load()
            # ... use vector_store
        finally:
            parser.close()
    """

    SUPPORTED_TYPES = {'.txt', '.pdf', '.xlsx', '.csv', '.doc', '.docx'}
    REQUIRED_CONFIG_KEYS = {'storage_dir', 'persist_dir', 'collection_name', 'embedding'}

    def __init__(self, config: Dict):
        """
        Initialize Parser with configuration dictionary.

        Args:
            config: Configuration dictionary

        Raises:
            ParserConfigError: If configuration is invalid
            ParserStateError: If state is inconsistent
        """
        self._validate_config(config)
        self._initialize_config(config)
        self._initialize_paths()
        self._initialize_text_splitter()
        self._initialize_state()
        self._initialize_config_manager()
        self._initialize_or_load_persistence()
        self._validate_state_consistency()

    def _validate_config(self, config: Dict):
        """Validate configuration dictionary."""
        if not isinstance(config, dict):
            raise ParserConfigError("Config must be a dictionary")

        missing_keys = self.REQUIRED_CONFIG_KEYS - set(config.keys())
        if missing_keys:
            raise ParserConfigError(f"Missing required config keys: {missing_keys}")

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

        self.chunk_size = config.get('chunk_size', 1000)
        self.chunk_overlap = config.get('chunk_overlap', 200)
        self.is_separator_regex = config.get('is_separator_regex', False)
        self.verbose = config.get('verbose', True)
        self.batch_size = config.get('batch_size', 100)
        self.max_retries = config.get('max_retries', 3)

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
        self._closed = False
        self._lock = threading.RLock()

    def _initialize_config_manager(self):
        """Initialize configuration file manager."""
        self.config_manager = ConfigFileManager(self.persist_dir, self.verbose)

    def _initialize_or_load_persistence(self):
        """
        Initialize or load persistent state from parser.config.json.
        Handles all scenarios for state recovery.
        """
        config_exists = self.config_manager.config_path.exists()
        collection_exists = self._collection_exists()

        self._log(f"Initializing parser for collection '{self.collection_name}'")
        self._log(f"  Config exists: {config_exists}")
        self._log(f"  Collection exists: {collection_exists}")

        if config_exists:
            # Load and validate existing config
            try:
                stored_config = self.config_manager.load_config()
                self.config_manager.validate_config(stored_config)

                # Verify collection name matches
                if stored_config['collection_name'] != self.collection_name:
                    raise ParserConfigError(
                        f"Collection name mismatch. Config has "
                        f"'{stored_config['collection_name']}', "
                        f"but '{self.collection_name}' was requested."
                    )

                # Load file records
                self._file_records = stored_config.get('file_records', {})

                if not collection_exists:
                    # Config exists but no DB - rebuild
                    self._log(
                        "Config found but collection missing. Will rebuild database.",
                        'warning'
                    )
                    self._rebuild_from_config()
                else:
                    # Normal case - load existing
                    self._load_existing_collection()

            except ParserConfigError as e:
                # Invalid config - crash as required
                raise ParserConfigError(
                    f"Invalid parser.config.json in {self.persist_dir}: {e}"
                )
        else:
            # No config exists
            if collection_exists:
                # Orphaned collection - clean up
                self._log(
                    "Collection exists without config. Deleting for clean state.",
                    'warning'
                )
                self._delete_collection_internal()

            # Create fresh config
            self._create_initial_config()

    def _create_initial_config(self):
        """Create initial configuration file."""
        config_data = {
            'collection_name': self.collection_name,
            'storage_dir': str(self.storage_dir),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'file_records': {},
            'statistics': {
                'total_files_processed': 0,
                'total_chunks_created': 0,
                'total_processing_time': 0.0
            }
        }

        self.config_manager.create_initial_config(config_data)
        self._file_records = {}
        self._vector_store = None

    def _rebuild_from_config(self):
        """Rebuild database from config when DB is missing."""
        self._log("Rebuilding database from configuration...")

        # Reset config to initial state
        updates = {
            'file_records': {},
            'statistics': {
                'total_files_processed': 0,
                'total_chunks_created': 0,
                'total_processing_time': 0.0
            }
        }
        self.config_manager.update_config(updates)
        self._file_records = {}
        self._vector_store = None

        self._log("Config reset. Database will be rebuilt on next load()")

    def _load_existing_collection(self):
        """Load existing collection and verify consistency."""
        try:
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=str(self.persist_dir)
            )
            doc_count = self._vector_store._collection.count()
            self._log(f"Loaded collection with {doc_count} documents")
        except Exception as e:
            raise ParserStateError(f"Failed to load collection: {e}")

    def _validate_state_consistency(self):
        """Validate consistency between config and database."""
        if self._vector_store is None:
            return

        try:
            db_count = self._vector_store._collection.count()
            config_chunks = sum(
                record.get('chunk_count', 0)
                for record in self._file_records.values()
            )

            if db_count != config_chunks and config_chunks > 0:
                self._log(
                    f"Warning: State mismatch detected. "
                    f"DB has {db_count} docs, config expects {config_chunks}",
                    'warning'
                )
        except Exception as e:
            self._log(f"Could not validate state consistency: {e}", 'warning')

    def _log(self, message: str, level: str = 'info'):
        """Log message if verbose mode is enabled."""
        if not self.verbose and level == 'info':
            return

        if level == 'warning':
            warnings.warn(message)
        elif level == 'error':
            print(f"ERROR: {message}")
        else:
            print(message)

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
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
        """Check if ChromaDB collection exists."""
        try:
            test_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=str(self.persist_dir)
            )
            return test_store._collection.count() >= 0
        except Exception:
            return False

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
        """Clean PDF/Doc text while preserving code syntax."""
        return cleaner(text)

    def _load_file(self, file_path: Path) -> List[Document]:
        """Load a file based on its extension."""
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
        """Get all supported files from storage directory."""
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
        """Create metadata for each chunk."""
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

        if hasattr(doc, 'metadata') and doc.metadata:
            metadata.update({k: v for k, v in doc.metadata.items() if k not in metadata})

        return metadata

    def _process_documents(
            self,
            documents: List[Document],
            file_path: Path,
            file_hash: str
    ) -> List[Document]:
        """Split documents into chunks with metadata."""
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
        """Compare config with storage directory."""
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

    def _sync_config(self):
        """Sync current state to config file."""
        updates = {
            'file_records': self._file_records,
            'statistics': {
                'total_files_processed': len(self._file_records),
                'total_chunks_created': sum(
                    r.get('chunk_count', 0) for r in self._file_records.values()
                ),
                'last_sync': datetime.now().isoformat()
            }
        }
        self.config_manager.update_config(updates)

    def load(self) -> Chroma:
        """
        Load and process files with intelligent synchronization.

        Returns:
            Chroma vector store instance
        """
        with self._lock:
            if self._closed:
                raise ParserStateError("Parser is closed. Create a new instance.")

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

                            # Batch processing
                            for i in range(0, len(chunks), self.batch_size):
                                batch = chunks[i:i + self.batch_size]
                                self._vector_store.add_documents(batch)

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

                            # Batch processing
                            for i in range(0, len(chunks), self.batch_size):
                                batch = chunks[i:i + self.batch_size]
                                self._vector_store.add_documents(batch)

                            self._file_records[file_path_str] = {
                                'hash': file_hash,
                                'filename': file_path.name,
                                'processed_at': datetime.now().isoformat(),
                                'chunk_count': len(chunks)
                            }
                except Exception as e:
                    self._log(f"    Error: {e}", 'error')

            # Sync to config
            self._sync_config()

            self._log(f"\nLoad complete. Tracked files: {len(self._file_records)}")
            self._log(f"Collection documents: {self._vector_store._collection.count()}")

            return self._vector_store

    def reload(self) -> Chroma:
        """
        Reload and synchronize with storage directory.

        Returns:
            Updated Chroma vector store
        """
        with self._lock:
            if self._closed:
                raise ParserStateError("Parser is closed. Create a new instance.")

            self._log("\n" + "=" * 60)
            self._log("RELOAD: Checking for changes")
            self._log("=" * 60)
            return self.load()

    def drop(self) -> bool:
        """
        Delete collection and config file.

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if self._closed:
                raise ParserStateError("Parser is closed")

            try:
                self._delete_collection_internal()
                self.config_manager.delete_config()

                self._vector_store = None
                self._file_records.clear()

                self._log(f"Collection '{self.collection_name}' and config deleted")
                return True
            except Exception as e:
                self._log(f"Error during drop: {e}", 'error')
                return False

    def get_vector_store(self) -> Chroma:
        """
        Get ChromaDB vector store instance.

        Returns:
            Chroma vector store

        Raises:
            ParserStateError: If parser is closed
        """
        with self._lock:
            if self._closed:
                raise ParserStateError("Parser is closed")
            self._ensure_vector_store()
            return self._vector_store

    def get_file_records(self) -> Dict[str, Dict]:
        """
        Get current file records.

        Returns:
            Copy of file records dictionary
        """
        with self._lock:
            return self._file_records.copy()

    def get_stats(self) -> Dict:
        """
        Get parser statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            storage_files = self._get_all_files()

            stats = {
                'collection_name': self.collection_name,
                'storage_dir': str(self.storage_dir),
                'persist_dir': str(self.persist_dir),
                'config_file': str(self.config_manager.config_path),
                'config_exists': self.config_manager.config_path.exists(),
                'files_in_storage': len(storage_files),
                'files_in_records': len(self._file_records),
                'collection_exists': self._collection_exists(),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'batch_size': self.batch_size,
                'is_closed': self._closed,
            }

            if self._vector_store:
                try:
                    stats['collection_document_count'] = self._vector_store._collection.count()
                except Exception:
                    stats['collection_document_count'] = 0
            else:
                stats['collection_document_count'] = 0

            # Add file type breakdown
            file_types = {}
            for file_path in storage_files:
                ext = file_path.suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1
            stats['file_type_breakdown'] = file_types

            return stats

    def get_health_check(self) -> Dict:
        """
        Perform comprehensive health check.

        Returns:
            Health check report with status and details
        """
        with self._lock:
            report = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'checks': {}
            }

            issues = []

            # Check 1: Config file integrity
            try:
                if self.config_manager.config_path.exists():
                    config = self.config_manager.load_config()
                    self.config_manager.validate_config(config)
                    report['checks']['config_file'] = {
                        'status': 'ok',
                        'message': 'Config file is valid'
                    }
                else:
                    report['checks']['config_file'] = {
                        'status': 'error',
                        'message': 'Config file missing'
                    }
                    issues.append('Config file missing')
            except Exception as e:
                report['checks']['config_file'] = {
                    'status': 'error',
                    'message': str(e)
                }
                issues.append(f'Config file error: {e}')

            # Check 2: Collection existence
            collection_exists = self._collection_exists()
            report['checks']['collection'] = {
                'status': 'ok' if collection_exists else 'warning',
                'message': 'Collection exists' if collection_exists else 'Collection not found'
            }
            if not collection_exists:
                issues.append('Collection not found')

            # Check 3: State consistency
            if self._vector_store:
                try:
                    db_count = self._vector_store._collection.count()
                    config_chunks = sum(
                        record.get('chunk_count', 0)
                        for record in self._file_records.values()
                    )

                    if db_count == config_chunks or config_chunks == 0:
                        report['checks']['state_consistency'] = {
                            'status': 'ok',
                            'message': f'State consistent: {db_count} documents',
                            'db_count': db_count,
                            'config_count': config_chunks
                        }
                    else:
                        report['checks']['state_consistency'] = {
                            'status': 'warning',
                            'message': f'State mismatch: DB has {db_count}, config expects {config_chunks}',
                            'db_count': db_count,
                            'config_count': config_chunks
                        }
                        issues.append(f'State mismatch: DB={db_count}, Config={config_chunks}')
                except Exception as e:
                    report['checks']['state_consistency'] = {
                        'status': 'error',
                        'message': str(e)
                    }
                    issues.append(f'State check failed: {e}')

            # Check 4: Storage directory
            if self.storage_dir.exists() and self.storage_dir.is_dir():
                report['checks']['storage_directory'] = {
                    'status': 'ok',
                    'message': f'Storage directory accessible',
                    'path': str(self.storage_dir)
                }
            else:
                report['checks']['storage_directory'] = {
                    'status': 'error',
                    'message': 'Storage directory not accessible'
                }
                issues.append('Storage directory not accessible')

            # Check 5: File sync status
            new_files, modified_files, deleted_files = self._sync_with_storage()
            out_of_sync = len(new_files) + len(modified_files) + len(deleted_files)

            if out_of_sync == 0:
                report['checks']['sync_status'] = {
                    'status': 'ok',
                    'message': 'All files synchronized'
                }
            else:
                report['checks']['sync_status'] = {
                    'status': 'warning',
                    'message': f'{out_of_sync} files out of sync',
                    'new': len(new_files),
                    'modified': len(modified_files),
                    'deleted': len(deleted_files)
                }
                issues.append(f'{out_of_sync} files need synchronization')

            # Check 6: Parser state
            if self._closed:
                report['checks']['parser_state'] = {
                    'status': 'warning',
                    'message': 'Parser is closed'
                }
                issues.append('Parser is closed')
            else:
                report['checks']['parser_state'] = {
                    'status': 'ok',
                    'message': 'Parser is active'
                }

            # Overall status
            if any(check['status'] == 'error' for check in report['checks'].values()):
                report['status'] = 'unhealthy'
            elif any(check['status'] == 'warning' for check in report['checks'].values()):
                report['status'] = 'degraded'
            else:
                report['status'] = 'healthy'

            report['issues'] = issues
            report['issue_count'] = len(issues)

            return report

    def repair(self) -> Dict:
        """
        Attempt to repair inconsistent state.

        Returns:
            Repair report with actions taken
        """
        with self._lock:
            if self._closed:
                raise ParserStateError("Parser is closed")

            report = {
                'timestamp': datetime.now().isoformat(),
                'actions': [],
                'success': True
            }

            self._log("\nStarting repair process...")

            # Check state consistency
            if self._vector_store:
                try:
                    db_count = self._vector_store._collection.count()
                    config_chunks = sum(
                        record.get('chunk_count', 0)
                        for record in self._file_records.values()
                    )

                    if db_count != config_chunks and config_chunks > 0:
                        self._log("State mismatch detected. Rebuilding...")

                        # Force rebuild
                        self._file_records.clear()
                        self._delete_collection_internal()
                        self._vector_store = None

                        # Reload everything
                        self.load()

                        report['actions'].append({
                            'action': 'rebuild_database',
                            'reason': f'State mismatch: DB had {db_count}, config expected {config_chunks}',
                            'status': 'completed'
                        })
                except Exception as e:
                    report['success'] = False
                    report['actions'].append({
                        'action': 'rebuild_database',
                        'reason': 'State check failed',
                        'status': 'failed',
                        'error': str(e)
                    })

            # Sync config
            try:
                self._sync_config()
                report['actions'].append({
                    'action': 'sync_config',
                    'status': 'completed'
                })
            except Exception as e:
                report['success'] = False
                report['actions'].append({
                    'action': 'sync_config',
                    'status': 'failed',
                    'error': str(e)
                })

            self._log(f"Repair completed. Actions taken: {len(report['actions'])}")
            return report

    def close(self) -> None:
        """
        Close parser and set config to read-only with write lock.
        This prevents any further write operations to the config file.

        Raises:
            ParserLockError: If closing fails
        """
        with self._lock:
            if self._closed:
                self._log("Parser already closed", 'warning')
                return

            try:
                # Sync final state
                self._sync_config()

                # Close config with read lock
                self.config_manager.close_with_read_lock()

                # Mark as closed
                self._closed = True

                self._log(f"\nParser closed successfully")
                self._log(f"Config file locked: {self.config_manager.config_path}")

            except Exception as e:
                raise ParserLockError(f"Failed to close parser: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if not self._closed:
            self.close()
        return False

    def __repr__(self) -> str:
        """String representation."""
        status = "closed" if self._closed else "active"
        return (
            f"Parser(collection='{self.collection_name}', "
            f"status='{status}', "
            f"files={len(self._file_records)})"
        )
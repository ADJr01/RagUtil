import os
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader
)
from langchain.schema import Document


class Parser:
    """
    Smart document parser with intelligent chunking and ChromaDB integration.
    Supports .txt, .pdf, .xlsx, .csv, .doc, and .docx files.
    """

    SUPPORTED_TYPES = {'.txt', '.pdf', '.xlsx', '.csv', '.doc', '.docx'}

    def __init__(self, storage_dir: str, persist_dir: str, collection_name: str, embedding):
        """
        Initialize the Parser with storage and persistence configurations.

        Args:
            storage_dir: Directory containing files to parse
            persist_dir: Directory where ChromaDB persists its database
            collection_name: Name of the collection in ChromaDB
            embedding: Embedding function for ChromaDB
        """
        self.storage_dir = Path(storage_dir)
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding = embedding

        # Create directories if they don't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize text splitter with intelligent chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False
        )

        self._vector_store: Optional[Chroma] = None
        self._file_hashes: Dict[str, str] = {}

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file for change detection."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _load_file(self, file_path: Path) -> List[Document]:
        """
        Load a single file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            List of Document objects
        """
        ext = file_path.suffix.lower()

        try:
            if ext == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif ext == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif ext == '.xlsx':
                loader = UnstructuredExcelLoader(str(file_path), mode="elements")
            elif ext == '.csv':
                loader = CSVLoader(str(file_path), encoding='utf-8')
            elif ext == '.docx':
                loader = Docx2txtLoader(str(file_path))
            elif ext == '.doc':
                loader = UnstructuredWordDocumentLoader(str(file_path))
            else:
                return []

            return loader.load()
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def _get_all_files(self) -> List[Path]:
        """Get all supported files from storage directory."""
        files = []
        for ext in self.SUPPORTED_TYPES:
            files.extend(self.storage_dir.glob(f"*{ext}"))
        return sorted(files)

    def _create_smart_metadata(self, doc: Document, file_path: Path, chunk_idx: int, total_chunks: int) -> Dict:
        """
        Create intelligent metadata for each chunk.

        Args:
            doc: Original document
            file_path: Path to source file
            chunk_idx: Index of current chunk
            total_chunks: Total number of chunks for this document

        Returns:
            Dictionary with smart metadata
        """
        metadata = {
            # File information
            'source': str(file_path),
            'filename': file_path.name,
            'file_type': file_path.suffix.lower(),
            'file_size': file_path.stat().st_size,
            'file_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),

            # Chunk information
            'chunk_index': chunk_idx,
            'total_chunks': total_chunks,
            'chunk_position': f"{chunk_idx + 1}/{total_chunks}",
            'is_first_chunk': chunk_idx == 0,
            'is_last_chunk': chunk_idx == total_chunks - 1,

            # Content characteristics
            'char_count': len(doc.page_content),
            'word_count': len(doc.page_content.split()),

            # Processing information
            'processed_at': datetime.now().isoformat(),
            'file_hash': self._get_file_hash(file_path),
        }

        # Add original metadata if exists
        if hasattr(doc, 'metadata') and doc.metadata:
            metadata.update({k: v for k, v in doc.metadata.items() if k not in metadata})

        return metadata

    def _process_documents(self, documents: List[Document], file_path: Path) -> List[Document]:
        """
        Split documents into intelligent chunks with smart metadata.

        Args:
            documents: List of documents to process
            file_path: Path to source file

        Returns:
            List of chunked documents with metadata
        """
        # Combine all document content
        full_text = "\n\n".join([doc.page_content for doc in documents])

        # Split into intelligent chunks
        chunks = self.text_splitter.create_documents([full_text])
        total_chunks = len(chunks)

        # Add smart metadata to each chunk
        processed_docs = []
        for idx, chunk in enumerate(chunks):
            chunk.metadata = self._create_smart_metadata(chunk, file_path, idx, total_chunks)
            processed_docs.append(chunk)

        return processed_docs

    def load(self) -> Chroma:
        """
        Load and process all supported files from storage directory.
        Creates or updates the ChromaDB collection.

        Returns:
            Chroma vector store instance
        """
        print(f"Loading files from {self.storage_dir}...")

        all_chunks = []
        files = self._get_all_files()

        if not files:
            print("No supported files found in storage directory.")

        for file_path in files:
            print(f"Processing: {file_path.name}")

            # Load file
            documents = self._load_file(file_path)
            if not documents:
                continue

            # Process into intelligent chunks
            chunks = self._process_documents(documents, file_path)
            all_chunks.extend(chunks)

            # Store file hash for change detection
            self._file_hashes[str(file_path)] = self._get_file_hash(file_path)

        print(f"Total chunks created: {len(all_chunks)}")

        # Create or update vector store
        if all_chunks:
            self._vector_store = Chroma.from_documents(
                documents=all_chunks,
                embedding=self.embedding,
                collection_name=self.collection_name,
                persist_directory=str(self.persist_dir)
            )
            print(f"Collection '{self.collection_name}' created/updated successfully.")
        else:
            print("No documents to process.")
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=str(self.persist_dir)
            )

        return self._vector_store

    def reload(self) -> Chroma:
        """
        Reload files and update vector store based on changes.
        Detects new, modified, and deleted files.

        Returns:
            Updated Chroma vector store instance
        """
        print("Reloading and checking for changes...")

        current_files = {str(f): f for f in self._get_all_files()}
        previous_files = set(self._file_hashes.keys())

        # Detect changes
        new_files = set(current_files.keys()) - previous_files
        deleted_files = previous_files - set(current_files.keys())

        # Check for modified files
        modified_files = set()
        for file_path in previous_files & set(current_files.keys()):
            current_hash = self._get_file_hash(Path(file_path))
            if current_hash != self._file_hashes[file_path]:
                modified_files.add(file_path)

        print(f"New files: {len(new_files)}")
        print(f"Modified files: {len(modified_files)}")
        print(f"Deleted files: {len(deleted_files)}")

        if not (new_files or modified_files or deleted_files):
            print("No changes detected.")
            return self.get_vector_store()

        # Initialize vector store if not exists
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=str(self.persist_dir)
            )

        # Handle deleted files
        for file_path in deleted_files:
            print(f"Removing deleted file: {Path(file_path).name}")
            self._vector_store.delete(where={"source": file_path})
            del self._file_hashes[file_path]

        # Handle modified files (delete old, add new)
        for file_path in modified_files:
            print(f"Updating modified file: {Path(file_path).name}")
            self._vector_store.delete(where={"source": file_path})

            documents = self._load_file(Path(file_path))
            if documents:
                chunks = self._process_documents(documents, Path(file_path))
                self._vector_store.add_documents(chunks)
                self._file_hashes[file_path] = self._get_file_hash(Path(file_path))

        # Handle new files
        for file_path in new_files:
            print(f"Adding new file: {Path(file_path).name}")
            documents = self._load_file(Path(file_path))
            if documents:
                chunks = self._process_documents(documents, Path(file_path))
                self._vector_store.add_documents(chunks)
                self._file_hashes[file_path] = self._get_file_hash(Path(file_path))

        print("Reload completed successfully.")
        return self._vector_store

    def drop(self) -> bool:
        """
        Delete the current collection from ChromaDB.

        Returns:
            True if collection was deleted, False if it didn't exist
        """
        try:
            client = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=str(self.persist_dir)
            )
            client.delete_collection()
            self._vector_store = None
            self._file_hashes.clear()
            print(f"Collection '{self.collection_name}' deleted successfully.")
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False

    def get_vector_store(self) -> Chroma:
        """
        Get the ChromaDB vector store instance.

        Returns:
            Chroma vector store instance
        """
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding,
                persist_directory=str(self.persist_dir)
            )
        return self._vector_store



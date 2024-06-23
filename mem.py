
import hashlib
from typing import List, Optional

import numpy as np
def md5(string):
    # 计算字符串的MD5哈希值
    md5_hash = hashlib.md5("some string".encode()).hexdigest()
    return md5_hash

class BucketBase(object):
    def __init__(self):
        pass

    def read(self,key):
        pass

    def write(self,key,value):
        pass

    def dump(self,path):
        pass

    def load(self,path):
        pass

    def describe(self,):
        pass

    def slice(self,):
        pass

    def merge_bucket(self,):
        pass

    def summary(self,):
        pass

    def clear(self,):
        pass

    def sanitize(self,):
        pass

    def optimize(self,):
        pass

    def export_offload(self,):
        pass


class MemoryUnitBase(object,BucketBase):
    def __init__(self,):
        pass

    def read(self,key):
        pass

    def write(self,key,value):
        pass

    def auto_optimize(self,):
        pass

    def forward_read(self,key):
        pass

    def forward_write(self,key,value):
        pass

    def off_load(self,):
        pass


class KeyValueBucket(BucketBase):
    def __init__(self):
        super().__init__()
        self.storage = {}
        self.log = {}
        self.offload_rate = 0.1
        self.offload_buffer = None

    def _log_access(self, key, operation_type):
        """记录键的访问次数，区分读和写操作。"""
        if key not in self.log:
            self.log[key] = {'read': 0, 'write': 0, 'remove': 0}
        if operation_type in self.log[key]:
            self.log[key][operation_type] += 1

    def read(self, key):
        """返回给定键的值，并记录读取操作。"""
        self._log_access(key, 'read')
        return self.storage.get(key, None)

    def write(self, key, value):
        """写入或更新键值对，并记录写入操作。"""
        self._log_access(key, 'write')
        self.storage[key] = value

    def remove(self, key):
        """从桶中移除指定的键，并记录移除操作。"""
        if key in self.storage:
            self._log_access(key, 'remove')
            del self.storage[key]

    def clear(self):
        """清空桶中的所有数据，并重置日志。"""
        self.storage.clear()
        self.log.clear()

    def describe(self):
        """返回桶的描述信息，并包括操作日志。"""
        return f"KeyValueBucket with {len(self.storage)} items. Log: {self.log}"

    def summary(self):
        """返回桶中数据的简要摘要，包括最频繁访问的键。"""
        if not self.log:
            return "No items in bucket."
        most_accessed_key = max(self.log, key=lambda k: sum(self.log[k].values()))
        access_details = self.log[most_accessed_key]
        return f"Most accessed key: {most_accessed_key} with accesses: {access_details}"

    def sanitize(self):
        """清理数据，确保所有键值对满足某些条件，例如数据类型一致性。"""
        for key, value in self.storage.items():
            if value is None:
                self.remove(key)

    def optimize(self):
        self.offload_buffer = {}
        # Calculate the number of elements to offload
        num_offload = int(len(self.storage) * 0.1)

        # Sort the log based on access count
        sorted_log = sorted(self.log.items(), key=lambda item: item[1])

        # Identify the keys of the least accessed elements
        offload_keys = [item[0] for item in sorted_log[:num_offload]]

        # Move the least accessed elements from storage to offload_buffer
        for key in offload_keys:
            self.offload_buffer[key] = self.storage[key]
            self.storage.pop(key)
        

    def export_offload(self):
        """导出数据以进行外部存储或处理。"""
        return self.offload_buffer


def remove(vectorstore: FAISS, docstore_ids: Optional[List[str]]):
    """
    Function to remove documents from the vectorstore.
    
    Parameters
    ----------
    vectorstore : FAISS
        The vectorstore to remove documents from.
    docstore_ids : Optional[List[str]]
        The list of docstore ids to remove. If None, all documents are removed.
    
    Returns
    -------
    n_removed : int
        The number of documents removed.
    n_total : int
        The total number of documents in the vectorstore.
    
    Raises
    ------
    ValueError
        If there are duplicate ids in the list of ids to remove.
    """
    if docstore_ids is None:
        vectorstore.docstore = {}
        vectorstore.index_to_docstore_id = {}
        n_removed = vectorstore.index.ntotal
        n_total = vectorstore.index.ntotal
        vectorstore.index.reset()
        return n_removed, n_total
    set_ids = set(docstore_ids)
    if len(set_ids) != len(docstore_ids):
        raise ValueError("Duplicate ids in list of ids to remove.")
    index_ids = [
        i_id
        for i_id, d_id in vectorstore.index_to_docstore_id.items()
        if d_id in docstore_ids
    ]
    n_removed = len(index_ids)
    n_total = vectorstore.index.ntotal
    vectorstore.index.remove_ids(np.array(index_ids, dtype=np.int64))
    for i_id, d_id in zip(index_ids, docstore_ids):
        del vectorstore.docstore._dict[
            d_id
        ]  # remove the document from the docstore

        del vectorstore.index_to_docstore_id[
            i_id
        ]  # remove the index to docstore id mapping
    vectorstore.index_to_docstore_id = {
        i: d_id
        for i, d_id in enumerate(vectorstore.index_to_docstore_id.values())
    }
    return n_removed, n_total


from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

class FAISSVectorDBBucket(BucketBase):
    def __init__(self,):
        super().__init__()
        self.docs = []
        self.embeddings = OpenAIEmbeddings()
        self.FAISS = FAISS.from_documents(self.docs, OpenAIEmbeddings())
        self.log = {}
        self.offload_rate = 0.1
        self.offload_buffer = None
        self.topK = 3
        self.text_splitor = CharacterTextSplitter(chunk_size=150, chunk_overlap=15)

    def _log_access(self, key, operation_type):
        """记录键的访问次数，区分读和写操作。"""
        if key not in self.log:
            self.log[key] = {'read': 0, 'write': 0, 'remove': 0}
        if operation_type in self.log[key]:
            self.log[key][operation_type] += 1

    def read(self, key):
        """返回给定键的值，并记录读取操作。"""
        ret = self.FAISS.similarity_search(key)
        for doc in ret:
            self._log_access(doc.content, 'read')
        return ret

    def write(self, text):
        """写入或更新键值对，并记录写入操作。"""
        text_splited = self.text_splitor.split_text(text)
        self.FAISS.add_documents(text_splited)

    def remove(self, key):
        """从桶中移除指定的键，并记录移除操作。"""
        self.log.pop(key)
        remove(self.FAISS, [key])

    def clear(self):
        """清空桶中的所有数据，并重置日志。"""
        self.FAISS = FAISS.from_documents([], OpenAIEmbeddings())
        self.log.clear()

    def describe(self):
        """返回桶的描述信息，并包括操作日志。"""
        return f"KeyValueBucket with {len(self.storage)} items. Log: {self.log}"

    def summary(self):
        """返回桶中数据的简要摘要，包括最频繁访问的键。"""
        if not self.log:
            return "No items in bucket."
        most_accessed_key = max(self.log, key=lambda k: sum(self.log[k].values()))
        access_details = self.log[most_accessed_key]
        return f"Most accessed key: {most_accessed_key} with accesses: {access_details}"

    def sanitize(self):
        """清理数据，确保所有键值对满足某些条件，例如数据类型一致性。"""
        pass

    def optimize(self):
        self.offload_buffer = {}
        # Calculate the number of elements to offload
        num_offload = int(len(self.storage) * 0.1)

        # Sort the log based on access count
        sorted_log = sorted(self.log.items(), key=lambda item: item[1])

        # Identify the keys of the least accessed elements
        offload_keys = [item[0] for item in sorted_log[:num_offload]]

        # Move the least accessed elements from storage to offload_buffer
        for key in offload_keys:
            self.offload_buffer[key] = self.storage[key]
            self.remove(key)

    def export_offload(self):
        """导出数据以进行外部存储或处理。"""
        return self.offload_buffer
    
class MemoryLayer()

import asyncio
from ..extensions import client, embedding_function, logger
from ..config import DEV_COLLECTION, PROD_COLLECTION
import numpy as np
import jieba

async def get_book_data(book_id: str, is_test: bool = False):
    """获取书籍数据"""
    try:
        collection_name = DEV_COLLECTION if is_test else PROD_COLLECTION
        loop = asyncio.get_event_loop()
        
        collection = await loop.run_in_executor(
            None,
            lambda: client.get_collection(name=collection_name)
        )
        
        results = await loop.run_in_executor(
            None,
            lambda: collection.get(where={"book_id": book_id})
        )
        
        if not results['ids']:
            return None
            
        # 重建书籍结构
        book_data = {
            'book_id': book_id,
            'title': results['metadatas'][0].get('book_title', ''),
            'author': results['metadatas'][0].get('book_author', ''),
            'description': results['metadatas'][0].get('book_description', ''),
            'chapters': []
        }
        
        # 按章节组织数据
        chapter_dict = {}
        for doc, metadata in zip(results['documents'], results['metadatas']):
            chapter_number = metadata.get('chapter_number')
            if chapter_number not in chapter_dict:
                chapter_dict[chapter_number] = {
                    'chapter_number': chapter_number,
                    'chapter_title': metadata.get('chapter_title', ''),
                    'chapter_index': metadata.get('chapter_index', -1),
                    'paragraphs': []
                }
            
            chapter_dict[chapter_number]['paragraphs'].append({
                'text': doc,
                'paragraph_number': metadata.get('paragraph_number'),
                'section_index': metadata.get('section_index', -1),
                'section_title': metadata.get('section_title', '')
            })
        
        # 按章节序号排序
        book_data['chapters'] = [
            chapter_dict[num] for num in sorted(chapter_dict.keys())
        ]
        
        return book_data
        
    except Exception as e:
        logger.error(f"获取书籍数据失败: {str(e)}")
        raise

async def perform_rag_search(query_text: str, book_id: str, collection_name: str, top_k: int = 3):
    """执行 RAG 检索"""
    try:
        loop = asyncio.get_event_loop()
        
        collection = await loop.run_in_executor(
            None,
            lambda: client.get_collection(name=collection_name)
        )
        
        # 检查文档数量
        check_results = await loop.run_in_executor(
            None,
            lambda: collection.get(where={"book_id": book_id})
        )
        
        doc_count = len(check_results['ids']) if check_results['ids'] else 0
        
        if doc_count == 0:
            return None, {
                'code': 404,
                'message': f'未找到书籍ID: {book_id}'
            }
            
        # 生成查询向量
        query_embedding = await loop.run_in_executor(
            None,
            lambda: embedding_function([query_text])
        )
        
        # 执行向量查询
        results = await loop.run_in_executor(
            None,
            lambda: collection.query(
                query_embeddings=query_embedding,
                where={"book_id": book_id},
                n_results=min(top_k * 2, doc_count),
                include=['documents', 'metadatas', 'distances']
            )
        )
        
        # 处理结果
        formatted_results = process_search_results(
            query_text,
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0],
            top_k
        )
        
        return formatted_results, None
        
    except Exception as e:
        logger.error(f"RAG检索失败: {str(e)}")
        return None, {
            'code': 500,
            'error': str(e)
        }

def process_search_results(query_text, documents, metadatas, distances, top_k):
    """处理搜索结果"""
    keywords = list(jieba.cut_for_search(query_text))
    formatted_results = []
    
    for doc, metadata, distance in zip(documents, metadatas, distances):
        keyword_score = sum(1 for keyword in keywords if keyword in doc) / len(keywords) if keywords else 0
        
        if keyword_score < 0.5 and distance > 0.8:
            continue
            
        result = {
            'content': doc,
            'chapter_info': {
                'chapter_number': metadata.get('chapter_number', ''),
                'index': metadata.get('chapter_index', ''),
                'title': metadata.get('chapter_title', '')
            },
            'section_info': {
                'paragraph_number': metadata.get('paragraph_number', ''),
                'index': metadata.get('section_index', ''),
                'title': metadata.get('section_title', '')
            },
            'scores': {
                'keyword': keyword_score,
                'vector_distance': float(distance)
            }
        }
        formatted_results.append(result)
    
    formatted_results.sort(key=lambda x: x['scores']['keyword'], reverse=True)
    return formatted_results[:top_k] 
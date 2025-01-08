import asyncio
from ..extensions import client, embedding_function, logger
from ..config import DEV_COLLECTION, PROD_COLLECTION
import numpy as np
import time

async def process_json_for_chroma(json_content: dict, book_id: str, is_test: bool = False):
    """处理 JSON 内容并添加到 ChromaDB"""
    try:
        collection_name = DEV_COLLECTION if is_test else PROD_COLLECTION
        logger.info(f"使用集合: {collection_name}")
        
        loop = asyncio.get_event_loop()
        
        # 确保集合存在
        try:
            collection = await loop.run_in_executor(
                None,
                lambda: client.get_collection(name=collection_name)
            )
        except Exception as e:
            logger.info(f"集合 {collection_name} 不存在，创建新集合")
            collection = await loop.run_in_executor(
                None,
                lambda: client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_function,
                    metadata={"environment": "development" if is_test else "production"}
                )
            )
        
        # 检查并删除已存在的数据
        try:
            existing_data = await loop.run_in_executor(
                None,
                lambda: collection.get(where={"book_id": book_id})
            )
            if existing_data['ids']:
                await loop.run_in_executor(
                    None,
                    lambda: collection.delete(where={"book_id": book_id})
                )
                logger.info(f"已删除 book_id: {book_id} 的现有数据")
        except Exception as e:
            logger.error(f"查询/删除现有数据时出错: {str(e)}")
            raise
        
        # 处理新的文本内容
        documents = []    # 原始文本
        metadatas = []    # 元数据
        ids = []          # 文档ID
        
        # 处理章节
        for chapter in json_content.get('chapters', []):
            chapter_number = chapter.get('chapter_number')
            chapter_title = chapter.get('chapter_title', '')
            
            # 处理段落
            for paragraph in chapter.get('paragraphs', []):
                if paragraph.get('text'):  # 只处理有文本内容的段落
                    doc_id = f"{book_id}_c{chapter_number}_p{paragraph.get('paragraph_number')}"
                    documents.append(paragraph['text'])
                    
                    # 构建完整的元数据
                    metadata = {
                        'book_id': book_id,
                        'chapter_number': chapter_number,
                        'chapter_title': chapter_title,
                        'chapter_index': chapter.get('chapter_index', -1),
                        'paragraph_number': paragraph.get('paragraph_number'),
                        'section_index': paragraph.get('section_index', -1),
                        'section_title': paragraph.get('section_title', ''),
                        'type': 'paragraph',
                        'book_title': json_content.get('title', ''),
                        'book_author': json_content.get('author', ''),
                        'book_description': json_content.get('description', '')
                    }
                    
                    metadatas.append(metadata)
                    ids.append(doc_id)
        
        if not documents:
            logger.warning(f"未找到任何文本内容: {book_id}")
            return {
                'status': 'warning',
                'message': '未找到任何文本内容',
                'document_count': 0
            }
            
        # 批量生成向量并添加到数据库
        try:
            embeddings = await loop.run_in_executor(
                None,
                lambda: embedding_function(documents)
            )
            
            # 分批添加数据
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                end = min(i + batch_size, len(documents))
                await loop.run_in_executor(
                    None,
                    lambda: collection.add(
                        documents=documents[i:end],
                        metadatas=metadatas[i:end],
                        embeddings=embeddings[i:end],
                        ids=ids[i:end]
                    )
                )
                await asyncio.sleep(0.1)  # 添加短暂延迟
                
            return {
                'status': 'success',
                'message': '文档更新成功',
                'document_count': len(documents),
                'updated': True
            }
            
        except Exception as e:
            logger.error(f"批量生成向量失败: {str(e)}")
            raise
            
    except Exception as e:
        error_msg = f"处理 JSON 失败: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

async def check_and_fix_collection(collection_name):
    """检查并修复集合"""
    try:
        logger.info(f"开始检查并修复集合: {collection_name}")
        
        loop = asyncio.get_event_loop()
        
        # 1. 备份现有数据
        try:
            collection = await loop.run_in_executor(
                None,
                lambda: client.get_collection(name=collection_name)
            )
            all_data = await loop.run_in_executor(
                None,
                collection.get
            )
            doc_count = len(all_data['ids']) if all_data['ids'] else 0
            logger.info(f"当前集合包含 {doc_count} 条数据")
            
            if doc_count == 0:
                logger.warning(f"集合 {collection_name} 为空，无需修复")
                return True
                
            # 保存重要的数据字段
            backup_data = {
                'ids': all_data['ids'],
                'embeddings': all_data['embeddings'],
                'documents': all_data['documents'],
                'metadatas': all_data['metadatas']
            }
            logger.info("数据备份完成")
            
        except Exception as e:
            logger.error(f"备份数据失败: {str(e)}")
            return False
            
        # 2. 删除并重新创建集合
        try:
            await loop.run_in_executor(
                None,
                lambda: client.delete_collection(collection_name)
            )
            logger.info(f"已删除集合: {collection_name}")
            
            # 等待一小段时间确保删除操作完成
            await asyncio.sleep(0.1)
            
            new_collection = await loop.run_in_executor(
                None,
                lambda: client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_function,
                    metadata={"recreated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                )
            )
            logger.info(f"已创建新集合: {collection_name}")
            
        except Exception as e:
            logger.error(f"重新创建集合失败: {str(e)}")
            return False
            
        # 3. 分批重新添加数据
        try:
            batch_size = 50  # 减小批次大小
            total_batches = (doc_count + batch_size - 1) // batch_size
            
            for i in range(0, doc_count, batch_size):
                batch_end = min(i + batch_size, doc_count)
                logger.info(f"正在处理批次 {i//batch_size + 1}/{total_batches}")
                
                # 准备批次数据
                batch_ids = backup_data['ids'][i:batch_end]
                batch_embeddings = backup_data['embeddings'][i:batch_end]
                batch_documents = backup_data['documents'][i:batch_end]
                batch_metadatas = backup_data['metadatas'][i:batch_end]
                
                # 添加数据
                await loop.run_in_executor(
                    None,
                    lambda: new_collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        documents=batch_documents,
                        metadatas=batch_metadatas
                    )
                )
                
                # 添加短暂延迟
                await asyncio.sleep(0.1)
                
            logger.info(f"集合 {collection_name} 修复完成")
            return True
            
        except Exception as e:
            logger.error(f"重新添加数据失败: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"修复集合失败: {str(e)}")
        return False 
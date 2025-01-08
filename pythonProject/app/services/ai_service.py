import aiohttp
import json
from ..extensions import logger
from ..config import DEEPSEEK_API_KEY, DEEPSEEK_API_URL

async def stream_deepseek_api(prompt, conversation_id=None):
    """异步流式调用 DeepSeek API"""
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
            }
            
            data = {
                'model': 'deepseek-chat',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.7,
                'max_tokens': 2000,
                'stream': True
            }
            
            if conversation_id:
                data['conversation_id'] = conversation_id
                
            async with session.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=data,
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_text = ""  # 用于累积完整的响应
                async for line in response.content:
                    if line:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            json_str = line[6:].strip()
                            if json_str == '[DONE]':
                                continue
                            try:
                                chunk = json.loads(json_str)
                                if chunk['choices'][0].get('delta', {}).get('content'):
                                    response_text += chunk['choices'][0]['delta']['content']
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON解析错误: {str(e)}, 原始数据: {json_str}")
                                continue
                
                return response_text
                    
    except Exception as e:
        error_msg = f"API调用失败: {str(e)}"
        logger.error(error_msg)
        return ""

async def get_analysis(analysis_prompt):
    """分析问题与教材的关联性"""
    try:
        response = await stream_deepseek_api(analysis_prompt)
        logger.info(f"API原始响应: {response}")
        
        try:
            # 清理响应文本
            response = response.strip()
            
            # 尝试找到第一个 { 和最后一个 } 的位置
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                # 提取JSON部分
                json_str = response[start_idx:end_idx + 1]
                logger.info(f"提取的JSON字符串: {json_str}")
                
                # 解析JSON
                result = json.loads(json_str)
                
                # 验证必要的字段
                if not isinstance(result.get('has_relation'), bool):
                    result['has_relation'] = False
                if 'analysis' not in result:
                    result['analysis'] = ''
                if 'related_question' not in result:
                    result['related_question'] = ''
                    
                return result
            else:
                logger.error("未找到有效的JSON结构")
                return {
                    "has_relation": False,
                    "analysis": "响应格式错误",
                    "related_question": ""
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {str(e)}, 原始响应: {response}")
            # 尝试使用正则表达式提取关键信息
            import re
            
            # 尝试提取 has_relation
            has_relation = False
            if '"has_relation":\\s*true' in response.lower():
                has_relation = True
                
            # 尝试提取 analysis
            analysis_match = re.search('"analysis":\\s*"([^"]*)"', response)
            analysis = analysis_match.group(1) if analysis_match else "无法解析分析结果"
            
            # 尝试提取 related_question
            question_match = re.search('"related_question":\\s*"([^"]*)"', response)
            related_question = question_match.group(1) if question_match else ""
            
            return {
                "has_relation": has_relation,
                "analysis": analysis,
                "related_question": related_question
            }
            
    except Exception as e:
        logger.error(f"分析过程发生错误: {str(e)}")
        return {
            "has_relation": False,
            "analysis": f"发生错误: {str(e)}",
            "related_question": ""
        } 
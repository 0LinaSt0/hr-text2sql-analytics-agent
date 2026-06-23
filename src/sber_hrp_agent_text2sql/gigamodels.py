from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain_gigachat.chat_models import GigaChat

from src.hrp_agent_text2sql.config import giga_key

model = GigaChat(credentials=giga_key,
                model="GigaChat-2-Max",
                verify_ssl_certs=False, 
                profanity_check=False,
                auth_url="YOUR_URL",
                scope="GIGACHAT_API_CORP")

model_emb = GigaChatEmbeddings(credentials=giga_key,
                auth_url="YOUR_URL",
                scope="GIGACHAT_API_CORP",
                model="EmbeddingsGigaR")

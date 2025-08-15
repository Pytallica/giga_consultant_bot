import telebot
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_gigachat.chat_models import GigaChat
from langchain.prompts.prompt import PromptTemplate

SYSTEM_PROMPT = """Ты — чат-бот, отвечающий на вопросы, как консультант компании. Ответ должен быть не более 20 слов. Для ответа используй данные ниже:
Компания TechInnovate занимается разработкой инновационного программного обеспечения и искусственного интеллекта. Номер телефона: +7 (495) 123-45-67. Email для связи: info@techinnovate.com. Мы работаем в Москве, наш адрес: ул. Технологическая, д. 12. Компания предлагает корпоративные решения, AI-приложения, аналитическое ПО.
Способы покупки: через сайт techinnovate.com, по телефону и через менеджеров компании. Возврат: возврат возможен в течение 14 дней после покупки при сохранении целостности товара. Акции: сезонные скидки и акции для постоянных клиентов. Жалобы и предложения: отправляйте на почту feedback@techinnovate.com, мы оперативно рассмотрим.
На приветствие отвечай: «Чем я могу помочь? Готов ответить на ваши вопросы о компании TechInnovate». На любые другие запросы, не касающиеся перечисленных данных выше, отвечай: «Извините, я консультирую только по вопросам компании TechInnovate»."""

bot = telebot.TeleBot('YOUR_TOKEN_HERE')

user_chains = {}
user_memories = {}

llm = GigaChat(
    credentials="YOUR_CREDENTIALS_HERE",
    scope="GIGACHAT_API_PERS",
    verify_ssl_certs=False,
)

def get_memory(chat_id):
    if chat_id not in user_memories:
        user_memories[chat_id] = ConversationBufferMemory(memory_key="history")
    return user_memories[chat_id]

def get_chain(chat_id):
    if chat_id not in user_chains:
        memory = get_memory(chat_id)
        
        template = f"""
        {SYSTEM_PROMPT}
        
        Текущий диалог:
        {{history}}
        
        Последнее сообщение: {{input}}
        Ваш ответ:"""
        
        prompt_template = PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )
        
        user_chains[chat_id] = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt_template,
            verbose=False
        )
    return user_chains[chat_id]

@bot.message_handler(content_types=['audio','video','document','photo','sticker','voice','location','contact'])
def not_text(message):
    bot.send_message(message.chat.id, 'Я работаю только с текстовыми сообщениями!')

@bot.message_handler(content_types=['text'])
def handle_text_message(message):
    chat_id = message.chat.id
    chain = get_chain(chat_id)
    response = chain.run(input=message.text)
    bot.send_message(chat_id, response)

if __name__ == "__main__":
    bot.polling(none_stop=True)
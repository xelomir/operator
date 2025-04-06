# --- START OF FILE exceptions.py ---

# exceptions.py

class OperatorError(Exception):
    """
    Базовый класс для всех пользовательских исключений, генерируемых Operator'ом.
    Используется для отлова всех специфичных для оператора ошибок.
    """
    pass

class BrowserInteractionError(OperatorError):
    """
    Базовое исключение для ошибок, возникающих при взаимодействии с браузером
    через BrowserManager (например, ошибки Playwright, не найденные элементы и т.д.).
    """
    pass

class ElementNotFoundError(BrowserInteractionError):
    """
    Исключение, генерируемое, когда ожидаемый веб-элемент не может быть найден
    на странице или не соответствует требуемым условиям (например, не видим)
    даже после повторных попыток.
    """
    pass

class NavigationError(BrowserInteractionError):
     """
     Исключение, возникающее при ошибке во время навигации по URL
     (например, таймаут загрузки страницы, ошибка сети, недоступный URL).
     """
     pass

class WaitTimeoutError(BrowserInteractionError):
    """
    Исключение, возникающее, когда ожидание определенного состояния элемента
    (например, 'visible', 'hidden', 'text(...)') превышает установленный таймаут.
    Наследуется от BrowserInteractionError, так как связано с браузером.
    """
    pass

class FileOperationError(OperatorError):
     """
     Исключение для ошибок, связанных с операциями файловой системы в пределах
     рабочей директории (workspace) оператора (например, ошибка записи/чтения файла,
     попытка доступа за пределы workspace).
     """
     pass

class AgentError(OperatorError):
     """
     Базовое исключение для ошибок, связанных с работой AI-агента (Groq, Gemini и т.д.).
     Включает ошибки API, ошибки парсинга ответа, ошибки конфигурации агента.
     """
     pass

class AgentResponseError(AgentError):
    """
    Исключение, когда ответ от AI-агента некорректен или не может быть
    распарсен (например, отсутствует обязательная команда, неверный формат,
    ошибка парсинга аргументов).
    """
    pass

class AgentAPIError(AgentError):
    """
    Исключение для ошибок, возникающих при взаимодействии с API AI-агента
    (например, ошибки сети, ошибки аутентификации, rate limits, ошибки сервера API).
    """
    pass

# --- END OF FILE exceptions.py ---
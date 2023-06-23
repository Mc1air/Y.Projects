# Определение стоимости автомобилей

**Описание проекта**
    
Сервис по продаже автомобилей с пробегом «Не бит, не крашен» разрабатывает приложение, чтобы привлечь новых клиентов. В нём можно будет узнать рыночную стоимость своего автомобиля. 
Постройте модель, которая умеет её определять. В вашем распоряжении данные о технических характеристиках, комплектации и ценах других автомобилей.
    
Критерии, которые важны заказчику:

    качество предсказания;
    время обучения модели;
    время предсказания модели.
    
**Описание данных**
    
Данные находятся в файле /datasets/autos.csv.
    
Признаки

    DateCrawled — дата скачивания анкеты из базы
    VehicleType — тип автомобильного кузова
    RegistrationYear — год регистрации автомобиля
    Gearbox — тип коробки передач
    Power — мощность (л. с.)
    Model — модель автомобиля
    Kilometer — пробег (км)
    RegistrationMonth — месяц регистрации автомобиля
    FuelType — тип топлива
    Brand — марка автомобиля
    Repaired — была машина в ремонте или нет
    DateCreated — дата создания анкеты
    NumberOfPictures — количество фотографий автомобиля
    PostalCode — почтовый индекс владельца анкеты (пользователя)
    LastSeen — дата последней активности пользователя

Целевой признак

    Price — цена (евро)

**Цель исследования:**
    
Построить модель для определения стоимости автомобилей с наилучшим качеством предсказания, 
минимальной скоростью обучения и предсказания.

**Ход исследования:**

1. Изучение данных
2. Предобработка данных
3. Подготовка данных к обучению моделей
4. Обучение моделей
5. Анализ скорости обучения и предсказания, качества предсказания моделей
6. Выбор лучшей модели и проверка её на тестовой выборке
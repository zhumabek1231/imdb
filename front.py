import streamlit as st
import requests

st.title('прецказания по коментария ')

api_url = 'http://127.0.0.1:8000/predict'

text = st.text_area(
    'Введите текст комента',
    height=200
)

if st.button('Проверка'):
    if text.strip():

        try:
            response = requests.post(
                api_url,
                json={"word": text},
                timeout=10
            )

            if response.status_code == 200:

                result = response.json()

                st.success(
                    f'Категория: {result["class"]}'
                )

            else:
                st.error(
                    f'Ошибка API: {response.status_code}'
                )
                st.write(response.text)

        except requests.exceptions.RequestException as e:
            st.error(f'Ошибка подключения: {e}')

    else:
        st.warning('Введите текст!')
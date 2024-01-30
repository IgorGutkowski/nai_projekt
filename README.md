# nai_projekt
Nasza aplikacja służy do rozpoznawania emocji twarzy oraz zwracania wyniku komunikatem dźwiękowym, tak aby poprawić inkluzywnośc komunikacji niewerbalnej.

W projekcie używamy Amazon Rekognition oraz Google Vision, aby Amazon Rekognition działało należy podać zmienne środowiskowe: $env:AWS_SECRET_ACCESS_KEY oraz $env:AWS_ACCESS_KEY_ID, aby Google Vision działało należy uzupełnić plik api.json danymi z google cloud.

Aby uruchomić frontend należy wykonać komendy npm install oraz npm start, aby uruchomić serwer należy użyć komendy python -m flask run, skrypty do statystyk uruchamiamy python amazon.py oraz python gvision.py.
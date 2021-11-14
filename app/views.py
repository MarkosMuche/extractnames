
from flair.data import Sentence
from flair.models import SequenceTagger
from rest_framework.response import Response
from rest_framework.views import APIView

tagger = SequenceTagger.load('ner')


sentence = Sentence('George Washington went to Washington.')


class PredictView(APIView):

    def post(self, request):
        data = request.data
        text = data['text']

        sentence = Sentence(text)
        tagger.predict(sentence)

        outp = ''
        for entity in sentence.get_spans('ner'):
            outp += str(entity)

        return Response(outp, status=200)

    def get(self, request):

        return Response('please enter any textual data for the result')

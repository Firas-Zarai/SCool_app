from haystack.nodes import QuestionGenerator, FARMReader
from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.pipeline import QuestionAnswerGenerationPipeline

model = "deepset/roberta-base-squad2"
#
# generate questions and answers from the content
#
def QAGen(contenu):
  question_generator = QuestionGenerator()
  reader = FARMReader(model)
  texts = [
      {"content":contenu},
  ]
  document_store = InMemoryDocumentStore()
  document_store.write_documents(texts)
  qag_pipeline = QuestionAnswerGenerationPipeline(question_generator, reader)
  results = []
  for doc in document_store:
      results.append(qag_pipeline.run(documents=[doc]))
  returnedVal = []
  for doc in results:
    for qa_pair in doc["results"]:
      returnedVal.append({"question":qa_pair["query"],"answer":qa_pair["answers"][0].answer})
      #print("Question: {}\nAnswer: {}\n".format(qa_pair["query"], qa_pair["answers"][0].answer))
  return returnedVal

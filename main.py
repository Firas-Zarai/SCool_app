from haystack.nodes import QuestionGenerator

question_generator = QuestionGenerator(model_name_or_path="valhalla/t5-base-e2e-qg")

text = """
As someone who has both taught English as a foreign language and has tried learning languages as a student, 
I know that it’s important to find interesting things to read when practicing reading comprehension. 
The internet is of course a great source of material. 
However, one difficulty when attempting to study using material you find online is that it’s not always easy 
to test your understanding. In order to get some feedback, you either have to find a teacher who will quiz you, 
or instead use a textbook which has some pre-written questions and answers. But a teacher is not always on-hand, 
and using textbooks significantly limits the range of reading material you can use.
"""

print(question_generator.generate(text))
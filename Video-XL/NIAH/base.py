# import torch

# class ViLLMBaseModel(torch.nn.Module):
class ViLLMBaseModel():
    def __init__(self, model_path, device):
        super().__init__()
        self.device=device
        self.model_path=model_path
        self.model_name=''

    def forward(self, instruction, videos):
        return self.generate(instruction, videos)
    
    def generate(self, instruction, videos):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        raise NotImplementedError
    
    def generate_prompt(self, question):
        """
        question: (str) a string of retrieval question
        Return: (str) a string of constructed prompt
        """
        # prompt_template = (
        #     "Question: {question}\n"
        #     "Provide a concise and relevant response based on the content of the image. "
        #     "Avoid including information not present in the image or repeating details unnecessarily. "
        #     "Formulate your answer clearly and coherently.\n"
        #     "Answer: "
        # )
        prompt_template = (
            "Question: {question}\n"
            "Answer: "
        )
        prompt = prompt_template.format(question=question)
        
        return prompt
import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Import your existing RAG engine
from skb_graphrag import GraphRAG 


# --- METRIC SCHEMA ---
class MetricScore(BaseModel):
    score: float = Field(description="A specific score between 0.0 and 1.0 based on the rubric.")
    reasoning: str = Field(description="A concise explanation citing specific parts of the text to justify the score.")

class RAGEvaluator:
    def __init__(self):
        self.rag = GraphRAG()
        self.judge_llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
        self.structured_judge = self.judge_llm.with_structured_output(MetricScore)
        
        # Perplexity Model (Unchanged)
        print(" Loading Perplexity Model (SmolLM2-1.7B)...")
        self.ppl_model_id = "HuggingFaceTB/SmolLM2-1.7B"

        self.ppl_tokenizer = AutoTokenizer.from_pretrained(self.ppl_model_id)
        self.ppl_model = AutoModelForCausalLM.from_pretrained(self.ppl_model_id)

        # Fix for models that lack a default pad token
        if self.ppl_tokenizer.pad_token is None:
            self.ppl_tokenizer.pad_token = self.ppl_tokenizer.eos_token

        self.ppl_model.eval()
        #self.ppl_model_id = "gpt2"
        #self.ppl_tokenizer = GPT2TokenizerFast.from_pretrained(self.ppl_model_id)
        #self.ppl_model = GPT2LMHeadModel.from_pretrained(self.ppl_model_id)
        #self.ppl_model.eval()

    def _get_eval_chain(self, system_prompt):
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input_data}"),
        ])
        return prompt | self.structured_judge

    # --- METRIC 1: Context Relevance (Rubric) ---
    def eval_context_relevance(self, question, context):
        rubric = """
        You are a strict judge evaluating the relevance of retrieved graph data.
        
        RUBRIC:
        - 0.0: Context is completely unrelated (noise).
        - 0.25: Context contains keywords but refers to the wrong topic (e.g., 'Apple' fruit vs company).
        - 0.5: Context contains some relevant entities but misses the specific relationship needed.
        - 0.75: Context contains the answer but includes a lot of irrelevant noise/distractions.
        - 1.0: Context is purely signal, containing the exact entities and relationships needed.
        
        Assign a score representing how helpful this context is.
        """
        chain = self._get_eval_chain(rubric)
        return chain.invoke({"input_data": f"Question: {question}\nContext: {context}"})

    # --- METRIC 2: Groundedness (Rubric) ---
    def eval_groundedness(self, answer, context):
        rubric = """
        You are a strict judge detecting hallucinations.
        Does the Answer rely ONLY on the provided Context?
        
        RUBRIC:
        - 0.0: The answer is entirely hallucinated (facts not in context).
        - 0.25: The answer contradicts the context.
        - 0.5: The answer mixes facts from the context with major external knowledge.
        - 0.75: The answer is mostly grounded but adds minor details/adjectives not present in context.
        - 1.0: Every claim in the answer is directly supported by a triple or description in the context.
        """
        chain = self._get_eval_chain(rubric)
        return chain.invoke({"input_data": f"Answer: {answer}\nContext: {context}"})

    # --- METRIC 3: Answer Relevancy (Rubric) ---
    def eval_answer_relevance(self, question, answer):
        rubric = """
        You are a judge evaluating if the Answer addresses the User's Question.
        
        RUBRIC:
        - 0.0: Answer is "I don't know" or completely irrelevant.
        - 0.25: Answer addresses the topic but answers a different specific question.
        - 0.5: Answer is vague or overly general.
        - 0.75: Answer addresses the question but misses a nuance or specific detail requested.
        - 1.0: Answer is direct, comprehensive, and addresses all constraints of the question.
        """
        chain = self._get_eval_chain(rubric)
        return chain.invoke({"input_data": f"Question: {question}\nAnswer: {answer}"})

    # --- METRIC 4: Contextual Precision (Ordering Rubric) ---
    def eval_precision(self, context, ground_truth):
        rubric = """
        You are evaluating the ranking of retrieved information.
        Is the MOST relevant information located at the TOP of the Context?
        
        RUBRIC:
        - 0.0: Relevant info is missing.
        - 0.25: Relevant info is present but buried at the very bottom of a long context.
        - 0.5: Relevant info is in the middle of the context.
        - 0.75: Relevant info is near the top but preceded by one or two irrelevant facts.
        - 1.0: The very first sentence/fact in the context matches the ground truth perfectly.
        """
        chain = self._get_eval_chain(rubric)
        return chain.invoke({"input_data": f"Ground Truth: {ground_truth}\nContext: {context}"})

    # --- METRIC 5: Context Recall (Rubric) ---
    def eval_recall(self, context, ground_truth):
        rubric = """
        You are evaluating retrieval completeness.
        Does the retrieved Context contain the key facts present in the Ground Truth?
        
        RUBRIC:
        - 0.0: Key concept is completely missing.
        - 0.5: The specific entity is found, but the relationship/property described in Ground Truth is missing.
        - 0.75: The concept is present, but the context lacks the specific detail/nuance of the Ground Truth.
        - 1.0: The Context contains the full semantic meaning of the Ground Truth.
        """
        chain = self._get_eval_chain(rubric)
        return chain.invoke({"input_data": f"Ground Truth: {ground_truth}\nContext: {context}"})

    # --- METRIC 6: Perplexity (Updated for AutoModel) ---
    def calc_perplexity(self, text: str) -> float:
        """
        Calculates perplexity using sliding window approach.
        """
        if not text or len(text.strip()) == 0: return 0.0

        # Tokenize
        encodings = self.ppl_tokenizer(text, return_tensors="pt")

        # Dynamic context window check (SmolLM2 usually supports 2048 or 4096)
        # We default to 1024 to be safe and fast on CPU
        max_length = getattr(self.ppl_model.config, "max_position_embeddings", 2048)
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()

            # Mask the tokens that were already processed in the previous stride
            # so we don't double-count their probability contribution
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.ppl_model(input_ids, labels=target_ids)

                # Modern models might return NaN if the sequence is empty or fully masked
                if outputs.loss is not None and not torch.isnan(outputs.loss):
                    nlls.append(outputs.loss)

            prev_end_loc = end_loc
            if end_loc == seq_len: break

        if not nlls:
            return 0.0

        # Compute average negative log likelihood and convert to Perplexity
        ppl = torch.exp(torch.stack(nlls).mean())
        return float(ppl)

    def run_benchmark(self, test_data):
        results = []
        print(f"🚀 Starting Evaluation on {len(test_data)} questions...")
        
        for q, truth in tqdm(test_data):
            try:
                retrieved_data = self.rag.retrieve_graph_context(q)
                formatted_context = self.rag.format_context(retrieved_data)
                
                if not retrieved_data:
                    generated_answer = "No information found."
                    c_rel, c_prec, c_rec = 0.0, 0.0, 0.0
                    groundedness = 1.0 
                    ans_rel = 0.0
                    perplexity = 0.0
                else:
                    generated_answer = self.rag.generate_answer(q)
                    
                    # 2. LLM Judges with new Rubrics
                    c_rel = self.eval_context_relevance(q, formatted_context).score
                    groundedness = self.eval_groundedness(generated_answer, formatted_context).score
                    ans_rel = self.eval_answer_relevance(q, generated_answer).score
                    c_prec = self.eval_precision(formatted_context, truth).score
                    c_rec = self.eval_recall(formatted_context, truth).score
                    
                    perplexity = self.calc_perplexity(generated_answer)

                weighted_score = (c_rel + groundedness + ans_rel + c_prec + (c_rec * 1.5)) / 5.5

                results.append({
                    "Question": q[:20] + "...",
                    "Answer": generated_answer[:12] + "...",
                    "Context_Rel.": c_rel,
                    "Groundedness": groundedness,
                    "Answer_Rel.": ans_rel,
                    "Context_Prec.": c_prec,
                    "Context_Rec.": c_rec,
                    "Perplexity": round(perplexity, 2),
                    "Comp_Score": round(weighted_score, 2)
                })
            except Exception as e:
                print(f"⚠️ Error: {e}")
                continue

        return pd.DataFrame(results)

if __name__ == "__main__":
    # Example Test Data
    TINY_TEST = [
        ("What is Opportunity Cost?", "The loss of potential gain from other alternatives when one alternative is chosen.")
    ]
    ECON_TEST = [
        # --- ECONOMICS (Intro Level) ---
        ("Explain the concept of Opportunity Cost.", "Opportunity cost is the loss of potential gain from other alternatives when one alternative is chosen."),
        ("What happens to the equilibrium price if supply decreases while demand remains constant?", "The equilibrium price will increase."),
        ("Define Gross Domestic Product (GDP).", "GDP is the total monetary or market value of all finished goods and services produced within a country's borders in a specific time period."),
        ("What is the difference between Microeconomics and Macroeconomics?", "Microeconomics studies individual units (people, firms) while Macroeconomics studies the economy as a whole (inflation, GDP)."),
        ("How does an increase in interest rates affect inflation?", "Higher interest rates generally lower inflation by reducing consumer spending and business investment."),
        ("What describes a Market Monopoly?", "A market structure characterized by a single seller, selling a unique product in the market with no competition."),
        ("What is Price Elasticity of Demand?", "It measures the responsiveness of the quantity demanded of a good to a change in its price."),
        ("Explain the Law of Diminishing Marginal Utility.", "As a person increases consumption of a product, there is a decline in the marginal utility derived from each additional unit."),
        ("What constitutes Fiscal Policy?", "Fiscal policy is the use of government spending and taxation to influence the economy."),
        ("What is the invisible hand theory?", "A metaphor describing the unintended greater social benefits and public good brought about by individuals acting in their own self-interests."),]
    LAW_TEST = [
        # --- LAW (Intro / Torts / Contracts) ---
        ("What are the four elements of Negligence?", "Duty, Breach, Causation, and Damages."),
        ("What is 'Consideration' in contract law?", "Consideration is a benefit which must be bargained for between the parties, and is the essential reason for a party entering into a contract."),
        ("Explain the concept of Mens Rea.", "Mens Rea refers to the 'guilty mind' or criminal intent behind a crime."),
        ("What does the First Amendment protect?", "It protects freedom of speech, religion, press, assembly, and the right to petition the government."),
        ("What is the difference between a Tort and a Crime?", "A tort is a civil wrong causing harm to an individual, while a crime is a wrongful act against society/state punishable by law."),
        ("Explain the principle of Stare Decisis.", "The legal principle of determining points in litigation according to precedent."),
        ("What is Double Jeopardy?", "The prosecution of a person twice for the same offense, prohibited by the 5th Amendment."),
        ("What is a Plaintiff?", "The party who initiates a lawsuit before a court."),
        ("Define 'Habeas Corpus'.", "A writ requiring a person under arrest to be brought before a judge or into court to secure the person's release unless lawful grounds are shown for their detention."),
        ("What is the 'Statute of Frauds'?", "A law that requires certain types of contracts to be in writing and signed by the party to be charged."),
    ]
    PHYSICS_TEST = [
        # --- PHYSICS (Intro / Mechanics) ---
        ("State Newton's Second Law of Motion.", "Force equals mass times acceleration (F=ma)."),
        ("What is the difference between Speed and Velocity?", "Speed is a scalar quantity (magnitude only), while velocity is a vector quantity (magnitude and direction)."),
        ("Define Kinetic Energy.", "Kinetic energy is the energy that an object possesses due to its motion."),
        ("What does the First Law of Thermodynamics state?", "Energy cannot be created or destroyed, only altered in form (Conservation of Energy)."),
        ("Explain Ohm's Law.", "The current through a conductor between two points is directly proportional to the voltage across the two points (V=IR)."),
        ("What is the acceleration due to gravity on Earth?", "Approximately 9.8 meters per second squared."),
        ("Define Momentum.", "Momentum is the product of the mass and velocity of an object (p=mv)."),
        ("What is Friction?", "The resistance that one surface or object encounters when moving over another."),
        ("What is a Vector quantity?", "A quantity that has both magnitude and direction."),
        ("Explain the concept of Work in physics.", "Work is the measure of energy transfer that occurs when an object is moved over a distance by an external force.")
    ]


    evaluator = RAGEvaluator()
    try:
        df = evaluator.run_benchmark(TINY_TEST)
        print("\n\n --- ECON TEST REPORT ---")
        print(df.to_markdown())

        df = evaluator.run_benchmark(ECON_TEST)
        print("\n\n --- ECON TEST REPORT ---")
        print(df.to_markdown())

        df = evaluator.run_benchmark(LAW_TEST)
        print("\n\n  --- LAW TEST REPORT ---")
        print(df.to_markdown())
        
        df = evaluator.run_benchmark(PHYSICS_TEST)
        print("\n\n --- PHYSICS TEST REPORT ---")
        print(df.to_markdown())
 
 
    finally:
        evaluator.rag.close()

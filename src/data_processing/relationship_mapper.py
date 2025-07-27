# Relationship mapping functions 
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json # Required for json.loads when converting stringified lists
import pickle # Required for saving graph objects

class RelationshipMapper:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the RelationshipMapper with the DataFrame containing skills and engineered features.
        """
        self.df = df.copy()
        self.skill_names = self.df['skill_name'].tolist()
        self.skill_to_idx = {name: i for i, name in enumerate(self.skill_names)}

    def build_prerequisite_dependency_graph(self):
        """
        Builds a directed graph for prerequisite dependencies.
        Nodes are skills, edges represent 'requires'.
        """
        print("Building prerequisite dependency graph...")
        graph = nx.DiGraph()
        graph.add_nodes_from(self.skill_names)
        for _, row in self.df.iterrows():
            skill = row['skill_name']
            for prereq in row['prerequisites']:
                if prereq in self.skill_names:
                    graph.add_edge(prereq, skill, type='prerequisite')
                else:
                    if prereq not in graph:
                        graph.add_node(prereq, type='external_prerequisite')
                    graph.add_edge(prereq, skill, type='prerequisite_external')
        print("Prerequisite graph built.")
        return graph

    def create_skill_similarity_matrix(self):
        """
        Creates a skill similarity matrix based on shared attributes.
        Uses TF-IDF and Cosine Similarity for textual attributes.
        """
        print("Creating skill similarity matrix...")
        self.df['combined_text_features'] = self.df['category'] + " " + \
                                            self.df['skill_type'] + " " + \
                                            self.df['market_trend'] + " " + \
                                            self.df['industry_usage'].apply(lambda x: " ".join(x)) + " " + \
                                            self.df['complementary_skills'].apply(lambda x: " ".join(x))

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.df['combined_text_features'])

        similarity_matrix = cosine_similarity(tfidf_matrix)
        print("Skill similarity matrix created.")
        return similarity_matrix, self.skill_names

    def map_complementary_skill_networks(self):
        """
        Builds an undirected graph for complementary skill relationships.
        Nodes are skills, edges represent 'enhances'.
        """
        print("Mapping complementary skill networks...")
        graph = nx.Graph()
        graph.add_nodes_from(self.skill_names)
        for _, row in self.df.iterrows():
            skill = row['skill_name']
            for comp_skill in row['complementary_skills']:
                if comp_skill in self.skill_names:
                    graph.add_edge(skill, comp_skill, type='complementary')
                else:
                    if comp_skill not in graph:
                        graph.add_node(comp_skill, type='external_complementary')
                    graph.add_edge(skill, comp_skill, type='complementary_external')
        print("Complementary skill networks mapped.")
        return graph

    def generate_skill_to_industry_affinity_scores(self):
        """
        Generates affinity scores between skills and industries.
        """
        print("Generating skill-to-industry affinity scores...")
        all_industries = sorted(list(set(i for sublist in self.df['industry_usage'] for i in sublist)))
        affinity_matrix = pd.DataFrame(0, index=self.skill_names, columns=all_industries)

        for idx, row in self.df.iterrows():
            skill = row['skill_name']
            for industry in row['industry_usage']:
                affinity_matrix.loc[skill, industry] = 1
        print("Skill-to-industry affinity scores generated.")
        return affinity_matrix

    def create_category_hierarchy_structures(self):
        """
        Identifies and represents hierarchies within categories (if applicable).
        """
        print("Creating category hierarchy structures...")
        category_parents = {
            "AI / ML": "Programming & Technical Skills",
            "Data Science & Analytics": "Programming & Technical Skills",
            "Web Development": "Programming & Technical Skills",
            "Mobile Development": "Programming & Technical Skills",
            "DevOps": "Programming & Technical Skills",
            "Cybersecurity": "Programming & Technical Skills",
            "UI/UX": "Design",
            "Game Art & Animation": "Design",
            "Game Design & Theory": "Design",
            "Soft Skills": "Soft Skills & General Professionalism",
            "Methodology": "Business & Management",
            "Production & Business": "Business & Management",
            "Database": "Programming & Technical Skills",
            "Programming": "Programming & Technical Skills",
            "Tool": "Programming & Technical Skills",
            "Mathematics": "Foundational Knowledge", # Example
        }
        category_graph = nx.DiGraph()
        unique_categories = self.df['category'].unique()
        for cat in unique_categories:
            category_graph.add_node(cat, type='category')
        for child, parent in category_parents.items():
            if parent not in category_graph:
                category_graph.add_node(parent, type='category')
            category_graph.add_edge(parent, child, type='parent_of')

        print("Category hierarchy structures created (conceptual).")
        return category_graph

    def build_skill_progression_pathways(self, prerequisite_graph: nx.DiGraph):
        """
        Infers learning pathways based on prerequisite relationships.
        """
        print("Building skill progression pathways...")
        print("Skill progression pathways can be queried from the prerequisite graph.")
        return "Refer to prerequisite_graph for pathway queries"

    def map_all_relationships(self):
        """
        Executes all relationship mapping methods.
        """
        print("Mapping relationships...")
        self.prerequisite_graph = self.build_prerequisite_dependency_graph()
        self.skill_similarity_matrix, self.similarity_skill_names = self.create_skill_similarity_matrix()
        self.complementary_skills_graph = self.map_complementary_skill_networks()
        self.skill_industry_affinity = self.generate_skill_to_industry_affinity_scores()
        self.category_hierarchy = self.create_category_hierarchy_structures()
        self.skill_progression_info = self.build_skill_progression_pathways(self.prerequisite_graph)
        print("Relationship mapping complete.")

if __name__ == "__main__":
    # Load the output from your Feature Engineering step
    input_path = 'data/processed/skills_engineered_features.jsonl'
    print(f"Loading engineered features from: {input_path}")
    df_after_features = pd.read_json(input_path, lines=True)

    for col in ['prerequisites', 'complementary_skills', 'industry_usage']:
        if col in df_after_features.columns and df_after_features[col].apply(type).eq(str).any():
            df_after_features[col] = df_after_features[col].apply(json.loads)

    relationship_mapper = RelationshipMapper(df_after_features)
    relationship_mapper.map_all_relationships()

    print("\nPrerequisite Graph Nodes (first 5):", list(relationship_mapper.prerequisite_graph.nodes(data=True))[:5])
    print("Prerequisite Graph Edges (first 5):", list(relationship_mapper.prerequisite_graph.edges(data=True))[:5])
    print("\nSkill Similarity Matrix (first 3x3):\n", relationship_mapper.skill_similarity_matrix[:3,:3])
    print("\nSkill to Industry Affinity (first 5 rows):\n", relationship_mapper.skill_industry_affinity.head())
    print("\nCategory Hierarchy Graph Nodes:", list(relationship_mapper.category_hierarchy.nodes(data=True)))
    print("Category Hierarchy Graph Edges:", list(relationship_mapper.category_hierarchy.edges(data=True)))

    # --- Save the relationships ---
    # Save the prerequisite graph
    prereq_graph_path = "data/processed/prerequisite_graph.gml"
    nx.write_gml(relationship_mapper.prerequisite_graph, prereq_graph_path)
    print(f"Prerequisite graph saved to: {prereq_graph_path}")

    # Save the complementary skills graph
    comp_graph_path = "data/processed/complementary_skills_graph.gml"
    nx.write_gml(relationship_mapper.complementary_skills_graph, comp_graph_path)
    print(f"Complementary skills graph saved to: {comp_graph_path}")

    # Save the skill similarity matrix (as a DataFrame for easier saving)
    similarity_df = pd.DataFrame(relationship_mapper.skill_similarity_matrix,
                                 index=relationship_mapper.similarity_skill_names,
                                 columns=relationship_mapper.similarity_skill_names)
    similarity_matrix_path = "data/processed/skill_similarity_matrix.csv"
    similarity_df.to_csv(similarity_matrix_path)
    print(f"Skill similarity matrix saved to: {similarity_matrix_path}")

    # Save the skill to industry affinity matrix
    affinity_matrix_path = "data/processed/skill_industry_affinity.csv"
    relationship_mapper.skill_industry_affinity.to_csv(affinity_matrix_path)
    print(f"Skill to industry affinity matrix saved to: {affinity_matrix_path}")

    # Save the category hierarchy graph (can be pickled for full object, or saved as GML)
    category_hierarchy_path = "data/processed/category_hierarchy.gml"
    nx.write_gml(relationship_mapper.category_hierarchy, category_hierarchy_path)
    print(f"Category hierarchy graph saved to: {category_hierarchy_path}")
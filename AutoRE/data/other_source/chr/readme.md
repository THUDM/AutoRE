# CHemical Reactions dataset (CHR)
The CHR (Chemical Reactant-Product interactions) dataset is a distantly supervised dataset dealing with binary interactions between chemicals.

The dataset consists of 12,094 abstracts and their titles from PubMed (https://www.ncbi.nlm.nih.gov/pubmed/).
The annotation of Chemicals was realized through the back-end of the semantic faceted search engine Thalia (http://www.nactem.ac.uk/Thalia/).
Chemical compounds were selected from the annotated entities and aligned with the graph database Biochem4j (http://biochem4j.org). 
Biochem4j is a freely available database that integrates several resources such as UniProt, KEGG and NCBI Taxonomy.
If two chemical entities were identified as related in Biochem4j, they were considered as positive instances in the dataset, otherwise as negative.


### Format
The corpus is provided in the PubTator format as follows,
```
<PMID>|t|<TITLE>
<PMID>|a|<ABSTRACT>
<PMID> <START OFFSET 1> <LAST OFFSET 1> <MENTION 1> <TYPE 1> <IDENTIFIER 1>
<PMID> <START OFFSET 2> <LAST OFFSET 2> <MENTION 2> <TYPE 2> <IDENTIFIER 2>
...
```

### Information 
The dataset was evaluated in Sahu et al. (2019).
Each of the annotated mentions can be associated with more the one Knowledge Base id (KB id).
Relations between the same entity are existent in the corpus but can be ignored during processing. 
For each candidate chemical pair, both directions should be generated as chemicals can be either a reactant (first argument) or a product (second argument) in an interaction.
The dataset statistics can be found in the following table.

|                   | Train      | Dev        | Test       |
| -------------     | ---------- | ---------- | ---------- |
| Articles          | 7,298      | 1,182      | 3,614      |
| Chemical Mentions | 65,477     | 10,885     | 32,353     |
| Reactions         | 19,997     | 3,243      | 9,750      |


## References
Sunil K Sahu, Fenia Christopoulou, Makoto Miwa and Sophia Ananiadou. 2019. (In Press). Inter-sentence Relation Extraction with Document-level Graph Convolutional Neural Network. In Proceedings of ACL.

Axel J Soto, Piotr Przybyła and Sophia Ananiadou. 2018. Thalia: Semantic search engine for biomedical abstracts. Bioinformatics, 35(10): 1799–1801 

Swainston, Neil and Batista-Navarro, Riza and Carbonell, Pablo and Dobson, Paul D and Dunstan, Mark and Jervis, Adrian J and Vinaixa, Maria and Williams, Alan R and Ananiadou, Sophia and Faulon, Jean-Loup and others. 2017. biochem4j: Integrated and extensible biochemical knowledge through graph databases. PloS ONE, 12(7): e0179130 


## Licence
The annotations in the CHR dataset were created at the National Centre for Text Mining (NaCTeM), School of Computer Science, University of Manchester, UK. They are licensed under a Creative Commons Attribution 4.0 International Licence.

PLEASE ATTRIBUTE NaCTeM WHEN USING THE CORPUS, AND PLEASE CITE THE FOLLOWING ARTICLE:</b>

Sunil K Sahu, Fenia Christopoulou, Makoto Miwa and Sophia Ananiadou. 2019. Inter-sentence Relation Extraction with Document-level Graph Convolutional Neural Network. In Proceedings of ACL.




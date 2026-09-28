from app.models.ontology import (
    MappedOntologyConcept,
    OntologyContext,
)
from app.ontology.constraints import (
    build_constraints,
)
from app.ontology.ontology_loader import (
    load_ontology,
)


query = (
    "Can Example Antihypertensive Class "
    "be used during pregnancy?"
)


context = OntologyContext(
    query=query,
    extracted_concepts=[
        "Example Antihypertensive Class",
        "pregnancy",
    ],
    mapped_concepts=[
        MappedOntologyConcept(
            original_text=(
                "Example Antihypertensive Class"
            ),
            concept_name=(
                "ExampleAntihypertensiveClass"
            ),
            label=(
                "Example Antihypertensive Class"
            ),
            uri=(
                "https://medontoagent.example.org/"
                "ontology/"
                "ExampleAntihypertensiveClass"
            ),
        ),
        MappedOntologyConcept(
            original_text="pregnancy",
            concept_name="Pregnancy",
            label="Pregnancy",
            uri=(
                "https://medontoagent.example.org/"
                "ontology/Pregnancy"
            ),
        ),
    ],
    unknown_concepts=[],
    mapped_count=2,
    unknown_count=0,
)


graph = load_ontology()


result = build_constraints(
    query=query,
    ontology_context=context,
    graph=graph,
)


print(
    result.model_dump_json(
        indent=2
    )
)
from app.ontology.concept_mapper import (
    build_ontology_context,
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


graph = load_ontology()


context = build_ontology_context(
    query,
    graph=graph,
)


constraints = build_constraints(
    query=query,
    ontology_context=context,
    graph=graph,
)


print(
    constraints.model_dump_json(
        indent=2
    )
)
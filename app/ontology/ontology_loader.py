from pathlib import Path

from rdflib import Graph


class OntologyLoadError(RuntimeError):
    """
    Raised when the MedOntoAgent ontology cannot be
    safely loaded.
    """


DEFAULT_ONTOLOGY_PATH = Path(
    "ontology/medical_ontology.ttl"
)


def load_ontology(
    ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH,
) -> Graph:
    """
    Load the MedOntoAgent Turtle ontology into an RDFLib Graph.

    The ontology is a controlled application asset.
    Arbitrary user-supplied RDF is not loaded by this function.
    """
    path = Path(
        ontology_path
    )

    if not path.exists():
        raise OntologyLoadError(
            f"Ontology file does not exist: {path}"
        )

    if not path.is_file():
        raise OntologyLoadError(
            f"Ontology path is not a file: {path}"
        )

    if path.stat().st_size == 0:
        raise OntologyLoadError(
            f"Ontology file is empty: {path}"
        )

    graph = Graph()

    try:
        graph.parse(
            source=str(path),
            format="turtle",
        )

    except Exception as exc:
        raise OntologyLoadError(
            f"Failed to parse ontology: {path}"
        ) from exc

    if len(graph) == 0:
        raise OntologyLoadError(
            (
                "Ontology parsed successfully but "
                "contains no RDF triples."
            )
        )

    return graph
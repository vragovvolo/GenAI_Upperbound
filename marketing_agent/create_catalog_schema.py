from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

def create_catalog_and_schema():
    # Initialize the Databricks Workspace client
    # Note: This assumes you have authentication set up via environment variables
    # or Databricks CLI configuration
    workspace = WorkspaceClient()

    try:
        # Create the catalog
        catalog_info = workspace.catalogs.create(
            name="cjc",
            comment="Catalog for CJC projects"
        )
        print(f"Created catalog: {catalog_info.name}")

        # Create the schema (database)
        schema_info = workspace.schemas.create(
            name="marketing",
            catalog_name="cjc",
            comment="Schema for marketing related data"
        )
        print(f"Created schema: {schema_info.name} in catalog: {schema_info.catalog_name}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    create_catalog_and_schema() 
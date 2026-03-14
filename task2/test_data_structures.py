class BaseProcessor:
    def __init__(self, mode: str = "sync"):
        self.mode = mode

    def _internal_log(self, msg: str):
        print(f"[LOG]: {msg}")

class AnalysisEngine(BaseProcessor):
    """Main class for testing nested structures."""
    
    def __init__(self, data: list):
        super().__init__(mode="async")
        self.data = data

    @property
    def data_summary(self):
        return len(self.data)

    def execute(self):
        """Method containing a nested function for scoping tests."""
        def transform(x):
            # Inner function AST node: ast.FunctionDef inside a FunctionDef
            return x * 1.05

        self._internal_log("Starting execution")
        return [transform(d) for d in self.data]

    class Config:
        """Nested class: ast.ClassDef inside a ClassDef."""
        VERSION = "2.1.0"
        
        @staticmethod
        def get_build_info():
            return {"env": "prod", "id": 101}
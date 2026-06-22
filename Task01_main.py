from API_Group8 import BioreactorClient, USER, PASSWORD


def do_experiment(scale: str = "micro", T: float = 30.0, pH: float = 6.5,
                  F1: float = 0.5, F2: float = 0.5, F3: float = 0.5) -> dict:
    """Login to the API and run a bioreactor experiment.
    Diese Funktion kann später im Skript aufgerufen werden.
    """
    client = BioreactorClient()
    client.login(USER, PASSWORD)
    result = client.run(scale, T=T, pH=pH, F1=F1, F2=F2, F3=F3)
    return result


if __name__ == "__main__":
    # example usage of the API client
    data = do_experiment()
    print("API-connection successful:", data)



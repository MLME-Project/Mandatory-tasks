import pandas as pd
import importlib
import sys
import os
from datetime import datetime

# Stelle sicher, dass Python das Skript im aktuellen Ordner findet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importiere dein bestehendes BO-Skript (ohne die .py Endung)
import bo_loopv6

def run_multiple_bo(num_runs: int = 5, output_excel: str = "bo_10_runs_summary"):
    """
    Führt den BO-Loop mehrfach aus und speichert eine Zusammenfassung in Excel.
    """
    print(f"\n{'='*80}")
    print(f"STARTE {num_runs} UNABHÄNGIGE BO-DURCHLÄUFE")
    print(f"{'='*80}")
    
    results = []
    
    for i in range(1, num_runs + 1):
        print(f"\n\n{'*'*80}")
        print(f"*** STARTE DURCHLAUF {i} VON {num_runs} ***")
        print(f"{'*'*80}\n")
        
        # WICHTIG: Das Modul neu laden! 
        # Dadurch werden globale Objekte wie der Logger und das Budget zurückgesetzt.
        importlib.reload(bo_loopv6)
        
        try:
            # Rufe die main() deines Skripts auf
            fixed_params, final_yield, total_cost = bo_loopv6.main()
            
            # Ergebnisse für die Excel-Tabelle sammeln
            results.append({
                'Run': i,
                'T (°C)': fixed_params.get('T'),
                'pH': fixed_params.get('pH'),
                'F1': fixed_params.get('F1'),
                'F2': fixed_params.get('F2'),
                'F3': fixed_params.get('F3'),
                'Gesamtkosten (EUR)': total_cost,
                'Final Yield': final_yield
            })
            
            print(f"\n[ERFOLG] Durchlauf {i} beendet. Yield: {final_yield:.6f}, Kosten: {total_cost:,.0f} EUR")
            
        except Exception as e:
            print(f"\n[FEHLER] Durchlauf {i} abgebrochen: {e}")
            import traceback
            traceback.print_exc()
            
            # Fehlerhaften Run trotzdem in die Liste aufnehmen, damit die Zuordnung stimmt
            results.append({
                'Run': i,
                'T (°C)': None, 'pH': None, 'F1': None, 'F2': None, 'F3': None,
                'Gesamtkosten (EUR)': None,
                'Final Yield': None
            })

    # Erstelle einen DataFrame aus den gesammelten Ergebnissen
    df = pd.DataFrame(results)
    
    # Berechne statistische Zusammenfassungen für die erfolgreichen Runs (optional, aber hilfreich)
    df_erfolgreich = df.dropna()
    if not df_erfolgreich.empty:
        mean_yield = df_erfolgreich['Final Yield'].mean()
        mean_cost = df_erfolgreich['Gesamtkosten (EUR)'].mean()
        print(f"\n{'='*80}")
        print(f"ZUSAMMENFASSUNG ÜBER {len(df_erfolgreich)} ERFOLGREICHE RUNS")
        print(f"Durchschnittlicher Yield: {mean_yield:.6f}")
        print(f"Durchschnittliche Kosten: {mean_cost:,.0f} EUR")
        print(f"{'='*80}\n")
    
    # Füge einen Zeitstempel zum Dateinamen hinzu
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_filename = f"{output_excel}_{timestamp}.xlsx"
    
    # Speichere als Excel-Datei (.xlsx)
    df.to_excel(final_filename, index=False)
    print(f"Alle {num_runs} Durchläufe beendet. Excel-Log gespeichert unter: {final_filename}")

if __name__ == '__main__':
    # Startet den Loop 10 Mal
    run_multiple_bo(num_runs=5)
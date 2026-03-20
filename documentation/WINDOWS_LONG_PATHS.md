# Corriger l’erreur « chemin trop long » lors de l’installation de TensorFlow avec pip

Sous Windows, si vous voyez une erreur du type :

```text
OSError: [Errno 2] No such file or directory: '...\fault_injection_service_config_parser.h'
HINT: This error might have occurred since this system does not support Windows Long Path support enabled.
```

c’est que **Windows limite par défaut la longueur des chemins à 260 caractères** ; le chemin des fichiers extraits par TensorFlow dépasse cette limite.

## Solution : activer les chemins longs (long paths) sous Windows

### Méthode 1 : PowerShell (recommandée)

1. Ouvrir **PowerShell en tant qu’administrateur** (clic droit sur « Démarrer » → « Terminal (Admin) » ou « Windows PowerShell (Admin) »).
2. Exécuter :

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

3. **Redémarrer l’ordinateur** (ou au minimum fermer tous les terminaux et l’IDE puis rouvrir).
4. Revenir dans le répertoire du projet et réinstaller les dépendances :

```powershell
cd C:\Users\pc\Desktop\moderation-service\src\efficientnet_lite_gpu
pip install -r requirements.txt
```

### Méthode 2 : Stratégie de groupe (Windows Pro / Entreprise uniquement)

1. Appuyer sur `Win + R`, taper `gpedit.msc`, Entrée.
2. Aller dans : **Configuration ordinateur** → **Modèles d’administration** → **Système** → **Système de fichiers**.
3. Double-cliquer sur **« Activer les chemins longs Win32 »**, choisir **« Activé »**, OK.
4. Redémarrer puis exécuter à nouveau `pip install -r requirements.txt`.

### Méthode 3 : Installer uniquement icrawler (sans mettre à jour TensorFlow)

Si TensorFlow fonctionne déjà et que vous voulez seulement ajouter `icrawler` sans déclencher l’erreur de chemin long :

```powershell
pip install icrawler
```

Vous pourrez ensuite activer les chemins longs et mettre à jour TensorFlow si besoin.

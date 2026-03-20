#!/usr/bin/env python3
"""
Télécharge automatiquement un jeu d’images depuis Google Images, classé comme en entraînement (Train/ClassName/).

Utilisation :
  1. Configurer les catégories et mots-clés dans dataset_download_config.yaml.
  2. Depuis la racine du projet (efficientnet_lite_gpu) :
     python -m tools.fetch_google_dataset
  Ou avec un fichier de config :
     python -m tools.fetch_google_dataset --config tools/dataset_download_config.yaml

Dépendances : pip install icrawler pyyaml
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
import time
import warnings
from pathlib import Path

import yaml

# Extensions d'images reconnues pour le comptage et l'équilibrage
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

# Racine du projet : répertoire parent du dossier tools (efficientnet_lite_gpu)
ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_main_config(root: Path) -> dict:
    """Charge config.yaml pour récupérer dataset_dir, train_dir, etc."""
    cfg_path = root / "config.yaml"
    if not cfg_path.exists():
        return {
            "dataset_dir": "train/dataset",
            "train_dir": "Train",
            "test_dir": "Test",
        }
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    tc = data.get("train_config", {})
    return {
        "dataset_dir": tc.get("dataset_dir", "train/dataset"),
        "train_dir": tc.get("train_dir", "Train"),
        "test_dir": tc.get("test_dir", "Test"),
    }


def _load_download_config(config_path: Path) -> dict:
    """Charge la config de téléchargement (catégories et mots-clés)."""
    if not config_path.exists():
        raise FileNotFoundError(f"Fichier de config introuvable : {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_classes_from_existing_train(train_dir: Path) -> list[str]:
    """Lit les noms de catégories depuis les sous-dossiers de Train."""
    if not train_dir.exists():
        return []
    classes = []
    for p in train_dir.iterdir():
        if p.is_dir() and not p.name.startswith("."):
            classes.append(p.name)
    return sorted(classes)


def _sanitize_filename(name: str) -> str:
    """Supprime les caractères interdits dans les noms de fichiers."""
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip() or "image"


def _log(msg: str) -> None:
    """Affiche un message et vide le buffer pour affichage immédiat."""
    print(msg, flush=True)


def _count_images_in_dir(dir_path: Path) -> int:
    """Compte le nombre de fichiers image dans un dossier."""
    if not dir_path.is_dir():
        return 0
    return sum(
        1
        for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def _list_image_files(dir_path: Path) -> list[Path]:
    """Retourne la liste des fichiers image dans un dossier."""
    if not dir_path.is_dir():
        return []
    return [
        f
        for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]


# En-têtes type navigateur pour réduire les 403
_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Referer": "https://duckduckgo.com/",
}


def _download_image_url(url: str, filepath: Path, timeout: int = 12) -> bool:
    """Télécharge une image depuis une URL avec en-têtes navigateur. Retourne True si succès."""
    try:
        import requests
        r = requests.get(url, headers=_DOWNLOAD_HEADERS, timeout=timeout, stream=True)
        r.raise_for_status()
        content = r.content
        if len(content) < 500:
            return False
        # On dérive l'extension via Content-Type pour homogénéiser les fichiers.
        ct = (r.headers.get("Content-Type") or "").lower()
        if "jpeg" in ct or "jpg" in ct:
            suf = ".jpg"
        elif "png" in ct:
            suf = ".png"
        elif "webp" in ct:
            suf = ".webp"
        elif "gif" in ct:
            suf = ".gif"
        else:
            suf = Path(url).suffix or ".jpg"
            if suf.lower() not in IMAGE_EXTENSIONS:
                suf = ".jpg"
        out = filepath.with_suffix(suf) if filepath.suffix != suf else filepath
        out.write_bytes(content)
        return True
    except Exception:
        return False


def _get_ddgs():
    """Import DDGS (nouveau paquet ddgs ou ancien duckduckgo_search)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*renamed to.*ddgs.*", category=RuntimeWarning)
        try:
            from ddgs import DDGS
            return DDGS
        except ImportError:
            pass
        try:
            from duckduckgo_search import DDGS
            return DDGS
        except ImportError:
            pass
    return None


def _fetch_class_duckduckgo(
    keywords: str | list[str],
    class_dir: Path,
    max_num: int,
    delay_after_download: float = 0.5,
    delay_between_pages: float = 2.0,
    start_index: int = 0,
) -> int:
    """
    Récupère des images via DuckDuckGo (plusieurs mots-clés, plusieurs pages) et les enregistre dans class_dir.
    keywords : un mot-clé (str) ou une liste de mots-clés pour multiplier les URLs.
    start_index : numéro de départ pour les noms de fichiers (pour ne pas écraser les images existantes).
    Retourne le nombre d'images enregistrées.
    """
    DDGS = _get_ddgs()
    if DDGS is None:
        _log("  Installez ddgs : pip install ddgs  (ou duckduckgo-search)")
        return 0
    if isinstance(keywords, str):
        keywords = [keywords]
    keywords = [k.strip() for k in keywords if k and isinstance(k, str)]
    if not keywords:
        return 0
    # On collecte plus d'URLs que nécessaire pour compenser les liens invalides.
    urls = []
    seen = set()
    page_size = 100
    target_urls = min(max_num * 4, 5000)
    try:
        with DDGS() as ddgs:
            for ki, kw in enumerate(keywords):
                if len(urls) >= target_urls:
                    break
                _log(f"  [Recherche {ki+1}/{len(keywords)}] « {kw} »...")
                page = 1
                while len(urls) < target_urls:
                    try:
                        kwargs = {"max_results": page_size}
                        if page > 1:
                            kwargs["page"] = page
                        batch = list(ddgs.images(kw, **kwargs))
                    except TypeError:
                        batch = list(ddgs.images(kw, max_results=page_size))
                    if not batch:
                        break
                    for r in batch:
                        u = r.get("image") or r.get("url")
                        if u and isinstance(u, str) and u.startswith("http") and u not in seen:
                            seen.add(u)
                            urls.append(u)
                    _log(f"  → {len(urls)} URLs (page {page})")
                    if len(batch) < page_size:
                        break
                    page += 1
                    if len(urls) >= target_urls:
                        break
                    if delay_between_pages > 0:
                        time.sleep(delay_between_pages)
                if len(keywords) > 1 and delay_between_pages > 0:
                    time.sleep(delay_between_pages)
    except Exception as e:
        _log(f"  Erreur recherche DuckDuckGo : {e}")
    if not urls:
        return 0
    _log(f"  Téléchargement : {len(urls)} URLs → objectif {max_num} images...")
    saved = 0
    total_urls = len(urls)
    # Téléchargement progressif avec logs périodiques pour suivre l'avancement.
    for i, url in enumerate(urls):
        if saved >= max_num:
            break
        if (i + 1) % 100 == 0 or i == 0:
            _log(f"  → URL {i+1}/{total_urls} | enregistrées : {saved}/{max_num}")
        name = f"{start_index + saved + 1:04d}"
        filepath = class_dir / name
        if _download_image_url(url, filepath):
            saved += 1
            if saved % 20 == 0 or saved == max_num:
                _log(f"  ✓ {saved}/{max_num} images enregistrées")
        if delay_after_download > 0:
            time.sleep(delay_after_download)
    return saved


def _balance_classes(train_dir: Path, class_names: list[str], sanitize_fn) -> None:
    """
    Équilibre le nombre d'images par catégorie : supprime des images
    dans les catégories qui en ont trop pour que toutes aient le même nombre (le minimum).
    Pas de plafond : on garde le maximum possible en égalisant.
    """
    counts = {}
    for name in class_names:
        class_dir = train_dir / sanitize_fn(name)
        counts[name] = _count_images_in_dir(class_dir)

    if not counts:
        return

    _log("--- Avant équilibrage ---")
    for name in class_names:
        _log(f"  {name} : {counts[name]} images")
    min_count = min(counts.values())
    max_count = max(counts.values())
    if min_count == max_count and min_count > 0:
        _log("Déjà équilibré : toutes les catégories ont le même nombre d'images.")
        return
    if min_count <= 0:
        _log("Équilibrage ignoré : au moins une catégorie n'a aucune image. Complétez les catégories vides puis relancez.")
        return

    _log(f"--- Équilibrage : réduction à {min_count} images par catégorie ---")
    for name in class_names:
        class_dir = train_dir / sanitize_fn(name)
        n = counts[name]
        if n <= min_count:
            _log(f"  {name} : {n} images (inchangé)")
            continue
        to_remove = n - min_count
        files = _list_image_files(class_dir)
        random.shuffle(files)
        for f in files[:to_remove]:
            try:
                f.unlink()
            except OSError:
                pass
        _log(f"  {name} : {n} -> {min_count} images (supprimé {to_remove})")

    _log(f"Équilibrage terminé : chaque catégorie a maintenant {min_count} images.")


def run(
    root_dir: Path | None = None,
    download_config_path: Path | None = None,
    dry_run: bool = False,
) -> None:
    root_dir = root_dir or ROOT_DIR
    root_dir = Path(root_dir).resolve()
    os.chdir(root_dir)

    _log(f"Racine du projet : {root_dir}")

    main_cfg = _load_main_config(root_dir)
    dataset_root = root_dir / main_cfg["dataset_dir"]
    train_dir = dataset_root / main_cfg["train_dir"]

    # Chemin de la config de téléchargement : par défaut tools/dataset_download_config.yaml
    if download_config_path is None:
        download_config_path = root_dir / "tools" / "dataset_download_config.yaml"
    else:
        download_config_path = Path(download_config_path).resolve()
    _log(f"Config téléchargement : {download_config_path}")
    download_cfg = _load_download_config(download_config_path)

    search_keywords = download_cfg.get("search_keywords") or {}
    max_num_per_class = int(download_cfg.get("max_num_per_class", 1000))
    min_size = download_cfg.get("min_size")
    if isinstance(min_size, list) and len(min_size) >= 2:
        min_size = tuple(int(x) for x in min_size[:2])
    else:
        min_size = (200, 200)
    only_classes = download_cfg.get("only_classes")
    if only_classes is not None and not isinstance(only_classes, list):
        only_classes = [only_classes]

    # Déterminer les catégories : union de Train (dossiers existants) et de search_keywords
    existing_train = _get_classes_from_existing_train(train_dir)
    all_class_names = sorted(set(existing_train) | set(search_keywords.keys()))
    for c in all_class_names:
        if c not in search_keywords:
            search_keywords[c] = c
    classes_to_run = all_class_names
    if only_classes:
        classes_to_run = [c for c in classes_to_run if c in only_classes]

    if not classes_to_run:
        _log("Aucune catégorie à traiter. Vérifiez only_classes ou ajoutez des dossiers dans Train.")
        sys.exit(0)
    _log(f"Catégories reconnues (Train + config) : {len(classes_to_run)}")

    search_engine = (download_cfg.get("search_engine") or "duckduckgo").strip().lower()
    if search_engine not in ("duckduckgo", "bing", "google"):
        search_engine = "duckduckgo"
    _log(f"Moteur de recherche : {search_engine}")
    _log(f"Catégories à traiter : {len(classes_to_run)} (max {max_num_per_class} images/catégorie)")
    if dry_run:
        _log("--- Mode dry-run (aucun téléchargement) ---")

    if search_engine in ("bing", "google"):
        try:
            from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
        except ImportError:
            _log("Installez icrawler : pip install icrawler")
            sys.exit(1)
        logging.getLogger("icrawler.downloader").setLevel(logging.CRITICAL)
        _log("(Certaines images peuvent être ignorées si le site refuse le téléchargement. C'est normal.)")
    else:
        _log("(DuckDuckGo + téléchargement avec en-têtes navigateur pour limiter les 403.)")

    train_dir.mkdir(parents=True, exist_ok=True)

    delay_between_categories = max(0, float(download_cfg.get("delay_between_categories", 0)))
    delay_after_download = max(0.0, float(download_cfg.get("delay_after_download", 0.5)))
    delay_between_pages = max(0.0, float(download_cfg.get("delay_between_pages", 2.0)))
    if search_engine == "duckduckgo" and delay_between_categories > 0:
        _log(f"(Délai entre catégories : {delay_between_categories} s pour limiter le rate-limit.)")

    for idx, class_name in enumerate(classes_to_run):
        keywords_raw = search_keywords[class_name]
        if isinstance(keywords_raw, list):
            keywords_list = [str(k).strip() for k in keywords_raw if k]
            keyword_display = " | ".join(keywords_list[:5])
        else:
            keywords_list = [str(keywords_raw).strip() or class_name]
            keyword_display = keywords_list[0]

        class_dir = train_dir / _sanitize_filename(class_name)
        class_dir.mkdir(parents=True, exist_ok=True)

        if dry_run:
            _log(f"  [dry-run] {class_name} → « {keyword_display} » → {class_dir}")
            continue

        if search_engine == "duckduckgo" and idx > 0 and delay_between_categories > 0:
            d = int(delay_between_categories)
            _log(f"  Pause {d} s avant la catégorie {idx+1}/{len(classes_to_run)}...")
            for _ in range(d):
                time.sleep(1)
                if (_ + 1) % 5 == 0:
                    _log(f"  ... {d - (_ + 1)} s restantes")

        current_count = _count_images_in_dir(class_dir)
        need_to_fetch = max(0, max_num_per_class - current_count)
        if need_to_fetch == 0:
            _log(f"{class_name} : déjà {current_count} images (>= {max_num_per_class}), pas de téléchargement.")
            continue

        max_rounds = max(1, int(download_cfg.get("max_rounds_per_class", 5)))
        _log(f"  [{idx+1}/{len(classes_to_run)}] {class_name} (actuel : {current_count}, objectif {max_num_per_class}, max {max_rounds} tours)")
        try:
            if search_engine == "duckduckgo":
                round_num = 0
                while round_num < max_rounds:
                    current_count = _count_images_in_dir(class_dir)
                    need_to_fetch = max(0, max_num_per_class - current_count)
                    if need_to_fetch == 0:
                        _log(f"  ✓ {class_name} : objectif {max_num_per_class} atteint.")
                        break
                    round_num += 1
                    _log(f"  --- Tour {round_num}/{max_rounds} : {current_count} images, manque {need_to_fetch} ---")
                    n = _fetch_class_duckduckgo(
                        keywords_list,
                        class_dir,
                        need_to_fetch,
                        delay_after_download,
                        delay_between_pages,
                        start_index=current_count,
                    )
                    new_total = current_count + n
                    _log(f"  Enregistré : +{n} (total ~{new_total})")
                    if n == 0:
                        _log(f"  Aucune nouvelle image ce tour, passage à la catégorie suivante.")
                        break
                    if new_total >= max_num_per_class:
                        break
                    if round_num < max_rounds:
                        _log(f"  Pause 20 s avant tour {round_num+1}...")
                        time.sleep(20)
            elif search_engine == "bing":
                crawler = BingImageCrawler(
                    storage={"root_dir": str(class_dir)},
                    log_level=None,
                )
                filters = {"size": "large"} if min_size and min_size[0] >= 200 else None
                crawler.crawl(
                    keyword=keyword_display,
                    max_num=need_to_fetch,
                    filters=filters,
                )
            else:
                crawler = GoogleImageCrawler(
                    storage={"root_dir": str(class_dir)},
                    log_level=None,
                )
                crawler.crawl(
                    keyword=keyword_display,
                    max_num=need_to_fetch,
                    min_size=min_size,
                )
        except Exception as e:
            _log(f"  Avertissement : erreur pour la catégorie {class_name} : {e}")

    # Équilibrage : ramener chaque catégorie au même nombre d'images (le minimum)
    if not dry_run:
        if download_cfg.get("balance", True):
            _log("")
            _log("Équilibrage des catégories (tous les dossiers dans Train, même nombre d'images par classe)...")
            all_train_classes = _get_classes_from_existing_train(train_dir)
            _balance_classes(train_dir, all_train_classes, _sanitize_filename)
        else:
            _log("Équilibrage désactivé (balance: false). Affichage des effectifs :")
            all_train_classes = _get_classes_from_existing_train(train_dir)
            for name in all_train_classes:
                class_dir = train_dir / _sanitize_filename(name)
                n = _count_images_in_dir(class_dir)
                _log(f"  {name} : {n} images")
        _log("")
        _log("Tous les téléchargements ont été lancés.")


def main() -> None:
    print("fetch_google_dataset: démarrage...", flush=True)
    parser = argparse.ArgumentParser(description="Récupère des images Google par catégorie d’entraînement")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Fichier de config YAML (défaut : tools/dataset_download_config.yaml)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Racine du projet (défaut : répertoire parent de tools)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher uniquement les catégories et mots-clés, sans télécharger",
    )
    args = parser.parse_args()
    run(
        root_dir=args.root,
        download_config_path=args.config,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

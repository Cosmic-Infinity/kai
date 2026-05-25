[app]

# (str) Title of your application
title = kai Dashboard

# (str) Package name
package.name = kaidashboard

# (str) Package domain (needed for android packaging)
package.domain = org.cosmicinfinity

# (str) Source code where the main.py lives
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,jpeg,kv,json

# (list) List of inclusions using pattern matching
#source.include_patterns = assets/*,images/*

# (list) Source files to exclude (let empty to exclude nothing)
#source.exclude_exts = spec

# (list) List of directory to exclude (let empty to exclude nothing)
#source.exclude_dirs = bin, .git, .buildozer

# (str) Application versioning (method 1)
version = 1.0.0

# (list) Application requirements
# comma separated e.g. requirements = sqlite3,kivy
requirements = python3, kivy==2.3.1, kivymd==1.2.0, requests, paho-mqtt, pillow, urllib3, chardet, idna, certifi

# (str) Custom source folders for requirements
# It may be useful when requirement-src is used
#requirements.source.kivy = ../kivy

# (list) Garden requirements
#garden_requirements =

# (str) Presplash of the application
#presplash.filename = %(source.dir)s/images/presplash.png

# (str) Icon of the application
#icon.filename = %(source.dir)s/images/icon.png

# (str) Supported orientations (one of landscape, sensorLandscape, portrait or all)
orientation = portrait

# (list) List of service to declare
#services = NAME:ENTRYPOINT_TO_PY,NAME2:ENTRYPOINT2_TO_PY

#
# OSX Specific
#

#
# Android specific
#

# (bool) Indicate if the XML export should be enabled
#android.xml = False

# (list) Android permissions to add
android.permissions = INTERNET

# (int) Target Android API, should be as high as possible.
android.api = 34

# (int) Minimum API your APK will support.
android.minapi = 21

# (int) Android SDK version to use
#android.sdk = 34

# (str) Android NDK version to use
#android.ndk = 25b

# (bool) Use --private data storage (True) or --dir public storage (False)
#android.private_storage = True

# (str) Android NDK directory (if empty, it will be automatically downloaded.)
#android.ndk_path =

# (str) Android SDK directory (if empty, it will be automatically downloaded.)
#android.sdk_path = /mnt/c/Users/KIIT0001/AppData/Local/Android/Sdk

# (str) ANT directory (if empty, it will be automatically downloaded.)
#android.ant_path =

# (str) Android entry point, default is to use start.py
#android.entrypoint = org.kivy.android.PythonActivity

# (list) Pattern to exclude for shrink (remove unused files)
#android.shrink_exclude_patterns =

# (list) Directory to exclude for shrink
#android.shrink_exclude_dirs =

# (list) List of Java .jar files to add to the libs so that they are compiled into the APK.
#android.add_jars = foo.jar,bar.jar,libs/bar.jar

# (list) List of Java files to add to the android project (for custom java code)
#android.add_src =

# (list) Android AAR archives to add (used for android libraries)
#android.add_aars =

# (list) Gradle dependencies to add
#android.gradle_dependencies =

# (list) Packaging exclusions
#android.exclude_packages = lib/arm64-v8a/libpy*

# (list) Android architectures to build for, choices: armeabi-v7a, arm64-v8a, x86, x86_64
android.archs = arm64-v8a, armeabi-v7a

# (bool) Allow service library to be built
#android.meta_data =

# (list) Android library project to add (will be added to project.properties)
#android.library_references =

# (str) Logcat filter to use
android.logcat_filters = *:S python:D

# (bool) Copy library instead of making a lib-copy (only for old ndk tools)
#android.copy_libs = 1

# (str) The Android Gradle build directory
#android.gradle_build_dir =

# (str) Custom Gradle template
#android.gradle_template =

# (str) Custom AndroidManifest.xml template
#android.manifest_template =

# (list) Android extra-source directories
#android.html_custom_path =

# (str) Android entrypoint class name (for advanced custom launchers)
#android.entrypoint_classname = org.kivy.android.PythonActivity

# (str) Path to a custom whitelist file
#android.whitelist =

# (list) Android custom bootstrap (leave empty for default)
#android.bootstrap =

# (list) Android custom whitelist
#android.whitelist_src =

#
# Python for android (p4a) specific
#

# (str) python-for-android branch to use, default is master
#p4a.branch = master

# (str) Custom p4a directory or git URL
#p4a.source_dir =

# (str) Custom bootstrap to use for python-for-android
#p4a.bootstrap = sdl2

# (list) Extra python-for-android recipes to compile
#p4a.local_recipes =


[buildozer]

# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2

# (int) Display warning if buildozer is run as root (0 = false, 1 = true)
warn_on_root = 1

# (str) Path to buildozer work directory
#buildozer.workdir = %(source.dir)s/.buildozer

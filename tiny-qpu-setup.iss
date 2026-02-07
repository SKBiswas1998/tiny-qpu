; ═══════════════════════════════════════════════════════
; tiny-qpu Interactive Quantum Lab — Windows Installer
; Built with Inno Setup (https://jrsoftware.org/isinfo.php)
;
; To compile: iscc tiny-qpu-setup.iss
; Or open in Inno Setup Compiler GUI and click Build > Compile
; ═══════════════════════════════════════════════════════

#define MyAppName "tiny-qpu Quantum Lab"
#define MyAppVersion "2.0"
#define MyAppPublisher "tiny-qpu"
#define MyAppURL "https://github.com/SKBiswas1998/tiny-qpu"
#define MyAppExeName "tiny-qpu.exe"

[Setup]
AppId={{B8E3F2A1-7C4D-4E5F-9A8B-1D2E3F4A5B6C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\tiny-qpu
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=installer_output
OutputBaseFilename=tiny-qpu-setup
SetupIconFile=assets\tiny-qpu-logo.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
WizardImageFile=assets\installer_sidebar.bmp
WizardSmallImageFile=assets\installer_small.bmp
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "quicklaunchicon"; Description: "Create a &Quick Launch icon"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main executable
Source: "dist\tiny-qpu.exe"; DestDir: "{app}"; Flags: ignoreversion

; Documentation
Source: "tiny_qpu_manual.pdf"; DestDir: "{app}\docs"; Flags: ignoreversion skipifsourcedoesntexist

; Logo assets
Source: "assets\tiny-qpu-logo.ico"; DestDir: "{app}\assets"; Flags: ignoreversion
Source: "assets\tiny-qpu-logo.png"; DestDir: "{app}\assets"; Flags: ignoreversion skipifsourcedoesntexist
Source: "assets\tiny-qpu-logo.svg"; DestDir: "{app}\assets"; Flags: ignoreversion skipifsourcedoesntexist

; README
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
; Start Menu shortcuts
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\assets\tiny-qpu-logo.ico"; Comment: "Launch the Interactive Quantum Lab"
Name: "{group}\User Manual (PDF)"; Filename: "{app}\docs\tiny_qpu_manual.pdf"; Comment: "Open the user manual"; Flags: createonlyiffileexists
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"

; Desktop shortcut
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\assets\tiny-qpu-logo.ico"; Tasks: desktopicon; Comment: "Launch the Interactive Quantum Lab"

; Quick Launch
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\assets\tiny-qpu-logo.ico"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Messages]
WelcomeLabel1=Welcome to tiny-qpu
WelcomeLabel2=This will install [name/ver] on your computer.%n%nBuild quantum circuits, visualize quantum states, and explore quantum computing — all in your browser. No internet needed.%n%nClick Next to continue.
FinishedHeadingLabel=Installation Complete
FinishedLabel=tiny-qpu Quantum Lab has been installed on your computer. Click Finish to launch it.

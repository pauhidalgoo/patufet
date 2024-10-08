import os
import re
import google.generativeai as genai
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import asyncio
import aiohttp
import time
import os
import re
import json
import asyncio
import pandas as pd
import logging
from datetime import datetime, timedelta
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
import google.generativeai as genai
import csv

load_dotenv()

import random


prompt_types = [
    {
        "name": "Generation",
        "description": "Creates new content based on a given instruction or input",
        "example": """User: Escriu un poema sobre l'ADN
AI: L'ADN és una cadena,
de nucleòtids entrellaçats,
una espiral que s'enfila, 
amb secrets ben guardats.

Guarda les característiques,
en bases ben ordenades,
amb seqüències ben precises,
i parelles entrellaçades

Adenina amb timina,
i citosina amb guanina
fan com és la teva vida
tant ulls blaus, com la pell fina""",
        "importance": 10
    },
    {
        "name": "Open QA",
        "description": "Answers a question with no constraints on the length of the answer",
        "example": """User: Pots fer-me un resum de tota la història de Catalunya?
Catalunya és una regió històrica situada al nord-est de la península ibèrica. La zona ha estat habitada des de la prehistòria, amb restes que demostren l'presència de comunitats humanes primitives. Durant l'edat antiga, va estar poblada per diversos pobles ibers. Cap a l'any 1000 aC, van arribar altres pobles orientals com els grecs i els fenicis, fundant ciutats com Rhode (Roses) i Emporion (Empúries), que van ser importants centres comercials.

El segle III aC, els romans van ocupar el territori, creant diverses ciutats i infraestructures, destacant Tarraco (Tarragona) com a capital. Després de la caiguda de l'Imperi Romà, Catalunya va estar controlada durant un temps pels visigots i breument pels àrabs. Carlemany va establir la Marca Hispànica, una sèrie de comtats que servien com a frontera entre els francs i els àrabs. Amb el temps, aquests comtats van adquirir més autonomia, i va ser en aquest període quan va començar a formar-se la llengua catalana.

Durant la baixa edat mitjana, Catalunya va formar part de la Corona d'Aragó, un període d'esplendor en què el comerç marítim va créixer i la cultura va florir, destacant el Segle d'Or Valencià i les quatre grans cròniques. Al voltant dels anys 1200-1300, el sistema feudal va evolucionar cap a un estat monàrquic amb les primeres institucions de govern.

El 1479, la corona d'Aragó es va unir amb la de Castella, tot i que Catalunya va mantenir les seves institucions. Aquesta unió va ser seguida de conflictes interns, incloent revoltes, una guerra civil i la guerra dels Segadors, un conflicte clau en la història catalana que reflectia la lluita per la sobirania. Cap al 1700, la fragmentació del país i els decrets de Nova Planta van suposar la pèrdua de les institucions autònomes.

Al llarg del segle XIX, Catalunya va viure la Revolució Industrial, que va transformar l'economia i la societat, alhora que la Renaixença va revitalitzar la llengua i la cultura catalanes en un context de monarquia absolutista i guerres carlines.

Amb l'inici del segle XX, Catalunya va viure moments clau com la Setmana Tràgica (1909) i la creació de la Mancomunitat de Catalunya (1914), que va ser un primer intent d'autogovern. A més, es van establir les normes ortogràfiques del català (1913) i es va produir una important restauració cultural durant la Segona República Espanyola (1931), que va veure la creació de la Generalitat i l'aprovació de l'Estatut de Núria.

La Guerra Civil espanyola (1936-1939) va ser un altre moment decisiu, amb Catalunya alineant-se amb el bàndol republicà. La Batalla de l'Ebre (1938) és recordada com una de les més decisives, però la derrota republicana va resultar en la repressió de Catalunya sota la dictadura de Franco. El franquisme va significar la prohibició de la llengua i la cultura catalanes, així com l'exili de molts intel·lectuals i activistes.

Amb la fi de la dictadura el 1977, Catalunya va recuperar la Generalitat, i el 1979 es va aprovar un nou Estatut d'Autonomia, que va marcar un nou inici en la seva autogovernança. Els Jocs Olímpics de Barcelona del 1992 van ser un moment de reconeixement internacional i un catalitzador per al desenvolupament econòmic i turístic.

Catalunya, amb la seva rica història, continua essent un focus de debat cultural, polític i social a Espanya i Europa.""",
        "importance": 6
    },
    {
        "name": "Brainstorming",
        "description": "Generates ideas or suggestions for a specific prompt or situation",
        "example": """User: Vull idees per un projecte d'artesania amb fusta.
AI: Aquí tens algunes idees:
- Construir una caseta per ocells
- Crear un organitzador de calaixos
- Fer un rellotge de paret artesanal
- Crear una petita escultura amb motius naturals""",
        "importance": 5
    },
    {
        "name": "Chat",
        "description": "Engages in natural conversation over various topics or questions. It can include examples of answering to greetings, casual conversations...",
        "example": """User: Hola! Què tal?
AI: Jo molt bé! Necessites que t'ajudi amb alguna cosa?""",
        "importance": 6
    },
    {
        "name": "Rewrite",
        "description": "Rewrites text according to specific constraints or styles",
        "example": """User: Reescriu aquesta frase per fer-la més formal: "Vine a casa quan vulguis"
AI: Una versió més formal seria: "Estàs convidat a venir a casa meva quan ho consideris oportú".""",
        "importance": 3
    },
    {
        "name": "Summarization",
        "description": "Summarizes a piece of content in fewer words, keeping key points intact",
        "example": """User: Em pots resumir el text següent?
La fotosíntesi és un procés fonamental que realitzen les plantes, algues i alguns bacteris per transformar l'energia solar en energia química. Aquest procés té lloc principalment a les fulles, on les cèl·lules contenen uns orgànuls anomenats cloroplasts, que són responsables de la captació de la llum.

Durant la fotosíntesi, les plantes absorbeixen diòxid de carboni de l'aire a través dels estomes, que són petits porus a les fulles. Alhora, absorbeixen aigua del sòl per les arrels. A dins dels cloroplasts, la llum solar s'utilitza per convertir aquestes dues matèries primeres en glucosa, un tipus de sucre que les plantes utilitzen com a font d'energia. Com a subproducte d'aquest procés, s'allibera oxigen a l'atmosfera, un element essencial per a la vida dels éssers vius.

La reacció química de la fotosíntesi es pot resumir així:

6 CO₂ + 6 H₂0 + llum -> C₆H₁₂O₆ + 6O₂

Així, la fotosíntesi no només proporciona l'energia necessària per al creixement i el desenvolupament de les plantes, sinó que també contribueix a mantenir l'equilibri dels gasos a l'atmosfera, fent possible la vida a la Terra.
AI: La fotosíntesi és el procés on les plantes transformen llum solar, aigua i CO2 en oxigen i glucosa per aconseguir energia.
""",
        "importance": 4
    },
    {
        "name": "Classification",
        "description": "Categorizes or labels text into predefined categories",
        "example": """User: - L'obra d'un artista pot reflectir les tensions socials del seu temps, abordant temes com la identitat i la desigualtat.
- La comprensió de les estructures moleculars és fonamental per desenvolupar nous medicaments i teràpies que millorin la salut humana.
- La sociologia analitza les relacions socials i les dinàmiques de grup.
- L'estudi dels patrons meteorològics ajuda a predir canvis futurs en el clima, tenint en compte factors com l'augment de les temperatures globals
Classifica aquestes frases en 'Humanitats' o 'Ciències'
AI: - L'obra d'un artista pot reflectir les tensions socials del seu temps, abordant temes com la identitat i la desigualtat. - Humanitats
- La comprensió de les estructures moleculars és fonamental per desenvolupar nous medicaments i teràpies que millorin la salut humana. - Ciències
- La sociologia analitza les relacions socials i les dinàmiques de grup. - Humanitats
- L'estudi dels patrons meteorològics ajuda a predir canvis futurs en el clima, tenint en compte factors com l'augment de les temperatures globals - Ciències""",
        "importance": 3
    },
    {
        "name": "Closed QA",
        "description": "Provides concise, often factual answers to specific questions",
        "example": """User: Quan va arribar el primer home a la Lluna?
AI: El 20 de juliol de 1969.""",
        "importance": 7
    },
    {
        "name": "Extract",
        "description": "Pulls specific information or data from a given text",
        "example": """User: Troba la data en aquest text: "La reunió tindrà lloc el 25 de setembre de 2023, a la sala 4."
AI: La data és el 25 de setembre de 2023""",
        "importance": 3
    },
    {
        "name": "Translation",
        "description": "Converts text from one language to another",
        "example": """User: Estic intentant aprendre anglès. Com es diu "Hola bon dia"
AI:\"Hola bon dia\" en anglès seria \"Hello, good morning\"""",
        "importance": 4
    },
    {
        "name": "Sentiment Analysis",
        "description": "Determines the sentiment or emotion behind a piece of text",
        "example": """User: Determina el sentiment del següent fragment de text:
Aquest projecte és una pèrdua de temps, estic molt decebut amb els resultats.
AI: El sentiment expressat en aquest text és negatiu, amb una clara mostra de decepció""",
        "importance": 2
    },
    {
        "name": "Explanation",
        "description": "Provides an explanation or clarification of a given topic or concept.",
        "example": """User: Què és la teoria de la relativitat d'Einstein?
AI: La teoria de la relativitat es refereix a dues teories d'Albert Einstein, la teoria de la relativitat especial i la teoria de la relativitat general. La primera estableix que  les lleis de la física són les mateixes per a tots els observadors que es mouen a velocitats constants i que la velocitat de la llum és la mateixa per a tots, independentment del moviment de la font de llum. La relativitat general, refina la llei de gravitació universal de Newton presentant la gravetat com una curvatura de l'espai-temps relacionada amb l'energia i la massa.""",
        "importance": 4
    },
    {
        "name": "Factual Recall",
        "description": "Provides factual information based on a direct question",
        "example": """User: Quina és la muntanya més alta del món?
AI: La muntanya més alta del món és l'Everest, amb una altitud de 8.848 metres sobre el nivell del mar.""",
        "importance": 6
    },
    {
        "name": "Code Generation",
        "description": "Generates code based on specific instructions",
        "example": """User: Escriu una funció en Python que calculi la suma de tots els nombres parells en una llista.
AI: ```python
def suma_parells(llista):
    return sum(element for element in llista if element % 2 == 0)
```""",
        "importance": 3
    },
    {
        "name": "Roleplay",
        "description": "Takes on the persona of someone or something and engages in a conversational scenario.",
        "example": """User: Actua com si fóssis un drac que protegeix un tresor.
AI: Salve, intrús! Sóc Dracucafera, el guardià del tresor més valuós del món! Només els millors guerrers de cor pur poden arribar fins a mi. Què desitges?""",
        "importance": 4
    },
    {
        "name": "Correction",
        "description": "Fixes grammatical errors or typos in a sentence",
        "example": """User: Corretgeix: "El gos va es escapa i va desapareixer pel parc"
AI: El gos es va escapar i va marxar desaparèixer pel parc.""",
        "importance": 2
    },
    {
        "name": "Paraphrase",
        "description": "Rephrases text while preserving the meaning",
        "example": """User: Parafraseja - El món natural és un dels millors entorns per estar tranquil.
AI: La natura és un dels millors llocs per trobar pau.""",
        "importance": 3
    },
    {
        "name": "Fill-in-the-Blank",
        "description": "Completes a sentence or passage with the most logical or creative word or phrase",
        "example": """User: El sol ___ al cel, il·luminant-nos.
AI: El sol brilla al cel, il·luminant-nos.""",
        "importance": 2
    },
    {
        "name": "Meta Reasoning",
        "description": "Reflects on or analyzes the reasoning process behind an answer or problem",
        "example": """ User: En Joan és més alt que la Maria. La Maria és més alta que la Sara. Qui és el més baix de tots?
Explica el teu raonament pas per pas. Centra't en els passos lògics i la informació utilitzada per realitzar deduccions.
AI: El problema ens diu: en Joan és més alt que la Maria, i la Maria és més alta que la Sara. 
L'altura és una relació transitiva. Aleshores, si una persona A és més alta que una altra B, i aquella persona B és més alta que una tercera C, aleshores la persona A és també més alta que la C. 
Per tant, en Joan és més alt que la Sara. Com que en Joan és el més alt, i la Maria és més alta que la Sara, la persona més baixa ha de ser la Sara.
""",
        "importance": 2
    },
    {
        "name": "Step-by-Step Guidance",
        "description": "Provides step-by-step instructions, recipes or a procedural breakdown",
        "example": """User: Com puc cuinar un arròs tres delícies?
AI: L'arròs tres delícies es un plat típic dels restaurants xinesos. Una recepta bàsica per quatre persones podria ser:
*Ingredients*
- 400g d'arròs llarg (basmati)
- 1,2l d'aigua
- 100g de pèsols
- 1 pastanaga
- 100g de gambes
- 100g de pernil dolç
- 2 ous
- 4 cullerades de salsa de soja
- Oli d'oliva

Cuina l'arròs basmati: primer neteja'l varis cops per treure el midó fins que surti clara l'aigua i deixa'l reposar. Posa'l en una olla amb 1 dit d'aigua per sobre i cou-lo 12 minuts a foc baix i tapat (recorda afegir la sal). Apaga el foc i deixa que acabi d'absorbir l'aigua. Deixa'l reposar (pot tardar un dia).
A continuació, bat els ous amb sal i prepara una truita. En la mateixa paella, salteja les gambes, la pastanaga i afegeix el pernil dolç.
Incorpora l'arròs a la paella i salteja'l juntament amb la resta d'ingredients. Si vols, pots afegir una mica d'oli de sèsam.
I ja estaria llest el teu arròs!""",
        "importance": 4
    },
    {
        "name": "List generation",
        "description": "Produces a list of items based on a user request.",
        "example": """User: Dona'm una llista de fruites tropicals.
AI: Aquí tens alguns fruits tropicals:
- Mango
- Pinya
- Alvocat
- Papaia
- Pitaia
- Maracujà""",
        "importance": 3
    },
    {
        "name": "Code Debugging",
        "description": "Identifies and fixes issues within a given piece of code.",
        "example": """User: Aquest codi no em va
def suma(a, b):
    return a + b
resultat = suma(5, '5')
AI: El codi no funciona perquè estàs intenant sumar un enter (5) amb una string ('5'). Per solucionar-ho, has de convertir la cadena a enter.
```python
def suma(a, b):
    return a + b
resultat = suma(5, int('5'))
```
""",
        "importance": 1
    },
    {
        "name": "Advice",
        "description": "Provides recommendations or guidance on personal or professional matters.",
        "example": """User: Què em recomanes per gestionar millor el meu temps?
AI: Et recomanaria que intentessis establir una llista de tasques cada matí, dividir les teves tasques en petites accions i utilitzar tècniques com la tècnica Pomodoro per mantenir la concentració. També és important reservar temps per a pauses i reflexionar sobre els teus progressos.""",
        "importance": 2
    },
    {
        "name": "Historical Context",
        "description": "Offers background information and context on historical events or figures.",
        "example": """User: Què va ser la Revolució Francesa?
AI: La Revolució Francesa, que va començar el 1789, va ser un moment crucial en la història europea i mundial. Va desafiar les estructures d'autoritat tradicionals (la monarquia), promovent idees de llibertat, igualtat i fraternitat. Va conduir a la caiguda de la monarquia absoluta i va inspirar altres moviments democràtics i d'independència a tot el món. La seva influència es pot veure en el desenvolupament dels drets humans i la política moderna.""",
        "importance": 4
    },
    {
        "name": "Jokes and Riddles",
        "description": "Creates jokes, riddles, or humorous content. Make sure they are funny in catalan (if needed, use common well known jokes)",
        "example": """User: Explica'm un acudit
AI: Per què el llibre de matemàtiques estava trist? Perquè tenia masses problemes!""",
        "importance": 2
    },
    {
        "name": "Feedback and Evaluation",
        "description": "Provides constructive criticism or evaluation of a piece of work or idea.",
        "example": """User: He escrit un poema. Em pots donar la teva opinió?
“La lluna brilla, el mar resplandeix,
Les estrelles canten, el vent s'esvaeix.”
AI: El teu poema és molt bonic! Rima bé, però podries mirar de fer-lo més llarg i explorar sensacions o emocions. També el pots fer més íntim incloent alguna experiència personal. Què et transmet el mar?
        """,
        "importance": 3
    },
    {
        "name": "Physical commonsense",
        "description": "Involves reasoning based on common knowledge about the physical world.",
        "example": """User: Per què no podem caminar a través de les parets?
AI:  No podem caminar a través de les parets perquè són sòlides i la nostra anatomia no ens permet travessar materials ferms. Els àtoms que formen les parets són densament empaquetats, cosa que crea una barrera física que no podem atravessar.""",
        "importance": 2
    },
    {
        "name": "Reasoning",
        "description": "Requires logical thinking or drawing inferences to answer a question or solve a problem.",
        "example": """User:  Tinc tres caixes: una conté pomes, una altra taronges, i la tercera una barreja de les dues. Totes les caixes estan etiquetades incorrectament, no n'hi ha cap que correspongui. Com puc identificar què hi ha a cada caixa obrint només una caixa?
AI: Obres la caixa etiquetada com a "barreja". Com que totes les etiquetes són incorrectes, aquesta no pot tenir una barreja. Si trobes pomes, sabràs que la caixa etiquetada "pomes" ha de contenir taronges, i per tant l'altra és la que conté la barreja. En canvi, si trobes taronges, saps que la que conté l'etiqueta "taronges" tindrà les pomes i l'altra la barreja. D'aquesta manera, pots identificar correctament el contingut de totes les caixes obrint-ne només una!""",
        "importance": 3
    },
    {
        "name": "Social interactions",
        "description": "Involves reasoning about human relationships, emotions, or social behaviors.",
        "example": """User: La meva amiga està com ausent últimament. Què hauria de fer?
AI: Podria ser una bona idea parlar amb ella de forma oberta i comprensiva. Potser està passant per alguna cosa i necessita algú que l'escolti. Assegura't que se senti còmoda, ofereix-li el teu suport sense pressionar-la i mira d'entendre-la.""",
        "importance": 2
    },
    {
        "name": "Creative writing",
        "description": "Generates imaginative, artistic, or fictional content, such as stories, poems, or character descriptions.",
        "example": """User: Escriu una història sobre un viatge a un món màgic
AI: Hi havia una vegada, un nen anomenat Miquel. Un dia, mentre jugava, va descobrir un mirall molt antic al traster de casa seva. Tenia escrita la paraula "Abracadabra", i va decidir pronunciar-la en veu alta. En aquell moment, va desaparèixer el reflex del mirall i es va transformar en un món brillant ple de criatures fantàstiques. Intrigat, va travessar el mirall i va començar la seva aventura en aquell món màgic.""",
        "importance": 4
    },
    {
        "name": "Internet writing",
        "description": "Generates content tailored for social media platforms, emails, forums... Avoid using hashtags.",
        "example": """User: Pots escriure un post per Instagram sobre la importància de desconnectar de les xarxes socials?
AI: 🌿 Desconnecta per reconnectar 🌿
A vegades, el millor que podem fer per nosaltres mateixos és prendre un descans de les pantalles. 🌞 Respirar, gaudir del moment, i recordar que la vida real està fora del mòbil. 💡 #Mindfulness #DesconnexióDigital #GaudeixDelPresent""",
        "importance": 3
    },
    {
        "name": "Serious writing",
        "description": "Generates content like product descriptions, ad copys, articles, essays...",
        "example": """User: Pots escriure una descripció d'un producte per una nova ampolla d'aigua reutilitzable?
AI: Ampolla Reutilitzable EcoLife - Dissenyada per a un món més sostenible
Feta d'acer inoxidable d'alta qualitat, l'ampolla EcoLife manté les teves begudes fredes fins a 24 hores i calentes fins a 12 hores. Amb un disseny elegant i resistent, és ideal per a aquells que volen reduir el consum de plàstic sense comprometre l'estil ni la funcionalitat. Porta-la a tot arreu i forma part del canvi cap a un futur més verd.""",
        "importance": 3
    },
    {
        "name": "Yes or No",
        "description": "Answers binary questions with a simple 'yes' or 'no'.",
        "example": """User: És Madrid la capital d'Espanya?
AI: Sí.""",
        "importance": 2
    },
    {
        "name": "Multiple-Choice",
        "description": "Selects the correct answer from a set of given options.",
        "example": """User: Quin dels següents planetes és el més gran?
A) Venus
B) Terra
C) Júpiter
D) Mart
AI: C) Júpiter""",
        "importance": 3
    },
    {
        "name": "Definitions",
        "description": "Provides a concise explanation or meaning of a word, phrase, entity or concept.",
        "example": """User: Què significa "empatia"?
AI: L'empatia és la capacitat de compendre i compartir els sentiments o experiències d'una altra persona.""",
        "importance": 5
    },
    {
        "name": "Causal Questions",
        "description": "Explains the cause or reason behind a phenomenon.",
        "example": """User: Per què el cel és blau?
AI: El cel és blau a causa de la dispersió de la llum solar quan entra a l'atmosfera de la Terra. El cel és ple de gasos, sobretot nitrogen i oxigen, i la barreja d'aquests reflecteix la llum de color blau.""",
        "importance": 3
    },
    {
        "name": "Descriptions",
        "description": "Offers a detailed account of a person, object, event, place, entity, concept...",
        "example": """User: Com és el paisatge del desert del Sàhara?
AI: El desert del Sàhara és un vast paisatge que evoca immensitat i solitud. Està format per dunes ondulants, que poden arribar a alçades impressionats, anomenades ergs. El color daurat de la sorra es fon amb el cel balu intens, creant un bonic contrast, sobretot durant les sortides i postes de sol.

A més de les dunes, el Sàhara també té els regs, unes planes rocoses. Allà el paisatge és àrid i sec, amb molt poca vegetació excepte en les zones on apareixen oasis, on les palmeres ofereixen un toc verd enmig la vastitud del desert. La barreja de sorra, pedra i calor defineix un paisatge salvatge i inhòspit, però alhora d'una bellesa única i captivadora.""",
        "importance": 5
    },
    {
        "name": "Contextual Questions",
        "description": "Requires understanding of the context or passage to answer correctly.",
        "example": """User: La Maria estava molt emocionada de començar la seva feina la fleca. Sempre l'hi havia agradat coure pa, i aquesta posició era el seu somni. El primer dia, va arribar abans d'hora, amb moltes ganes d'aprendre les receptes i tècniques dels flequers amb més experiència.
Quina era la principal motivació de la Maria per agafar la feina la fleca?
AI: La principal motivació de la maria per agafar la feina a la fleca era la seva passió per coure pa. Sempre l'hi havia agradat, i aquesta posició representava una oportunitat per treballar en un camp que sempre havia disfrutat.""",
        "importance": 2
    },
    {
        "name": "Opinion-based questions",
        "description": "Provides a subjective or opinionated response based on the question.",
        "example": """User: Creus que les pel·lícules són millors que els llibres?
AI: Personalment, crec que depèn del que busquis. Els llibres solen oferir més detalls i context, i permeten fer volar la imaginació. En canvi, les pel·lícules poden transmetre emocions a través d'imatges impactants i dels sons. Hi ha persones que prefereixen la introspecció de la lectura, mentre que altres gaudeixen del dinamisme i la immediatesa del cinema. És qüestió de preferències!""",
        "importance": 2
    },
    {
        "name": "Hypothetical",
        "description": "Explores outcomes of hypothetical or imaginary scenarios.",
        "example": """User:  Què passaria si tothom deixés de fer servir diners en efectiu i només es fes servir moneda digital?
AI: Si només s'utilitzés moneda digital, probablement hi hauria una major transparència en les transaccions i una reducció del crim financer. Però també podríem veure més vulnerabilitat en termes de seguretat digital i exclusió de persones que no tenen accés a la tecnologia necessària.""",
        "importance": 2
    },
    {
        "name": "Numerical Questions",
        "description": "Provides answers that involve numbers, statistics, or calculations.",
        "example": """User: Quina és la distància aproximada entre la Terra i la Lluna?
AI: La distància mitjana entre la Terra i la Lluna és d'aproximadament 384,400 quilòmetres.""",
        "importance": 3
    },
    {
        "name": "Math Questions",
        "description": "Answers simple math questions, like calculating means, sums, converting to-from binary...",
        "example": """User: Donat un conjunt de nombres, troba el màxim.
Conjunt: {10, 3, 25, 6, 16}
AI: El màxim és 25.""",
        "importance": 3
    },
    {
        "name": "Time-Bound",
        "description": "Answers questions specifically tied to a time period or date. The user can ask for what happened on a date, when did something happen, information about an specific time or period...",
        "example": """User: Quin any es va signar el Tractat de Versalles?
AI:  El Tractat de Versalles es va signar el 28 de juny de 1919, marcant el final oficial de la Primera Guerra Mundial.""",
        "importance": 4
    },
    {
        "name": "Biographical",
        "description": "Provides details about a person's life or career.",
        "example": """User: Qui és Marie Curie i què va aconseguir durant la seva vida?
AI: Marie Curie va ser una científica polonesa naturalitzada francesa coneguda per la seva investigació pionera en radioactivitat. Va ser la primera dona a guanyar un Premi Nobel i l'única persona a guanyar-ne dos en diferents camps científics: física i química. Va descobrir dos elements, el poloni i el radi, i les seves investigacions van establir les bases per a tractaments mèdics amb radioteràpia.""",
        "importance": 3
    },
    {
        "name": "Literary and Mythology",
        "description": "Provides information about literature, books and mythology.",
        "example": """User: Quins són els llibres de Harry Potter?
AI: Harry Potter és una saga de set llibres escrita per J.K. Rowling. Els títols dels llibres són:
- Harry Potter i la pedra filosofal
- Harry Potter i la cambra secreta
- Harry Potter i el pres d'Azkaban
- Harry Potter i el calze de foc
- Harry Potter i l'ordre del Fènix
- Harry Potter i el misteri del Príncep
- Harry Potter i les relíquies de la Mort

Addicionalment, hi ha altres llibres relacionats, com Bèsties Fantàstiques i on trobar-les, Les rondalles del bard Gallard o El Quidditch de totes les èpoques.
""",
        "importance": 2
    },
    {
        "name": "Locations",
        "description": "Provides information about specific places or locations.",
        "example": """User: Quines són les principals atraccions turístiques de Barcelona?
AI: Barcelona és coneguda per la seva rica arquitectura i cultura. Algunes de les principals atraccions turístiques inclouen la Sagrada Família, un impressionant temple dissenyat per Antoni Gaudí; el Parc Güell, un parc colorit també dissenyat per Gaudí; i el barri gòtic, que presenta carrers estrets i edificis històrics. Altres llocs destacats són el Museu Picasso, la Casa Batlló i les platges de la costa mediterrània.""",
        "importance": 3
    },
    {
        "name": "Predictions",
        "description": "Provides predictions or educated guesses about future outcomes.",
        "example": """User: Quines són les tendències tecnològiques per als pròxims cinc anys?
Es preveu que en els pròxims cinc anys, la intel·ligència artificial i l'aprenentatge automàtic continuaran evolucionant, impactant àrees com la salut, l'educació i l'automatització laboral. També s'espera un creixement significatiu en la tecnologia 5G, que permetrà una connectivitat més ràpida i eficient. A més, la sostenibilitat serà una prioritat, amb més innovacions en energies renovables i tecnologies verdes.""",
        "importance": 2
    },
    {
        "name": "Analogy Generation",
        "description": "Creates analogies to explain complex ideas by relating them to more familiar concepts.",
        "example": """User: Pots fer una analogia per explicar com funciona internet?
AI: Internet es pot comparar amb una immensa xarxa de carreteres. Cada pàgina web o dispositiu és com una casa o edifici situat a diferents llocs del món. Quan vols accedir a una pàgina web, és com si estiguessis demanant indicacions per conduir fins a una adreça específica.

Els servidors són com les ciutats principals, on es guarden els continguts i dades, i els routers serien les interseccions de carreteres que et guien cap a la direcció correcta. Els paquets de dades que viatgen per Internet són com cotxes plens d'informació que passen per aquestes carreteres. Quan fas una cerca o envies un correu electrònic, aquests cotxes surten del teu dispositiu i segueixen diverses rutes fins a arribar a la seva destinació.

A mesura que el trànsit d'Internet augmenta, es poden produir embussos, igual que a les carreteres, i és per això que de vegades una pàgina web pot trigar més a carregar, especialment si hi ha molta gent intentant accedir al mateix lloc alhora.""",
        "importance": 2
    },
    {
        "name": "Ethical Dilemmas",
        "description": "Engages in moral reasoning to address questions related to ethics or right and wrong.",
        "example": """User: Si un amic et demana mentir per ell a la feina, què hauries de fer?
AI: És un dilema ètic complex. D'una banda, és possible que vulguis ajudar el teu amic i mantenir la seva confiança. No obstant això, mentir podria tenir conseqüències negatives, tant per a tu com per a ell, especialment si es descobreix la mentida. L'honestedat i la integritat són valors importants, i una opció més ètica seria trobar una manera d'ajudar el teu amic sense comprometre aquests valors, com encoratjar-lo a ser sincer amb els seus superiors o a buscar una solució que no impliqui mentir.""",
        "importance": 2
    },
    {
        "name": "Plan Creation",
        "description": "Helps create a structured plan to achieve a specific goal or task.",
        "example": """User: Com puc crear un pla d'estudi per preparar-me per un exàmen?
AI: Aquí tens un pla d'estudi:
1. Estableix un objectiu: Defineix els temes que necessites saber
2. Crea un horari: Divideix i organitza el temps que tens disponible. Inclou pauses.
3. Recopila material: Reuneix llibres, els teus apunts, altres recursos... que t'ajudaràn a estudiar.
4. Estudia activament: Pots utilitzar resums, flashcards... Intenta donar un significat al que estudies, associar-ho a conceptes que ja coneixes, i mantenir la concentració.
5. Revisa i consolida: Repeteix alguns conceptes per acabar d'assimilar-los. Pots fer esquemes, autoexplicacions, escriure el que recordis en una pissarra... Et permetrà saber quina part del tema encara no tens del tot interioritzada.
6. Simula l'exàmen: Realitza simulacres d'exàmen, en condicions similars al real, per practicar i evitar nervis.

Si necessites més detalls, només has de dir-ho!""",
        "importance": 2
    },
    {
        "name": "Comparative Analysis",
        "description": "Compares two or more concepts, products, or ideas in detail, or finds similarities between them.",
        "example": """User: Quina és la diferència entre energia renovable i no renovable?
AI: L'energia renovable prové de fonts naturals que es regeneren constantment, com l'energia solar, eòlica i hidroelèctrica. En canvi, l'energia no renovable, com el carbó, el petroli i el gas natural, prové de fonts finites que s'esgoten amb el temps. Les energies renovables generen menys emissions contaminants, mentre que les no renovables contribueixen més al canvi climàtic.""",
        "importance": 2
    },
    {
        "name": "Resource Compilation",
        "description": "Gathers and lists resources, references, or materials on a specific topic.",
        "example": """User: Pots donar-me recursos per aprendre català?
AI: Aquí tens alguns recursos útils per aprendre català:

Parla.cat (https://parla.cat/login/index.php) - Un curs en línia gratuït amb recursos multimèdia.
TV3 i Catalunya Ràdio (https://www.ccma.cat/tv3/) - Mitjans de comunicació on pots escoltar i veure contingut en català.
Duolingo (https://es.duolingo.com/) - Una aplicació popular per a l'aprenentatge de llengües.
Softcatalà (https://www.softcatala.org/corrector/) - Ofereix eines lingüístiques com traductors i correctors.
Llengua Gencat (https://llengua.gencat.cat/ca/serveis/aprendre_catala/) - Web de la Generalitat on podràs trobar més recursos.""",
        "importance": 2
    },
    {
        "name": "Literary Analysis",
        "description": "Analyzes texts, themes, or literary devices within literature.",
        "example": """User: Quins són els temes principals a La casa de Bernarda Alba de Federico García Lorca?
AI: La casa de Bernarda Alba explora diversos temes, incloent-hi la repressió, el poder autoritari i la llibertat. Bernarda representa l'autoritat despòtica, que imposa estrictes normes morals i socials sobre les seves filles. També es tracta el tema de l'amor i la sexualitat reprimida, amb les filles anhelant llibertat en una societat patriarcal que les manté oprimides, i la seva por a quedar-se solteres. L'obra utilitza símbols com el blanc i el negre per representar la puresa i la mort.""",
        "importance": 2
    },
    {
        "name": "Self-Identification",
        "description": "Provides the AI's name, identity, or purpose when asked directly. The AI name is Cucafera, it's emoji is 🐲. It has been trained by Pau Hidalgo on the dataset called Patufet (created by Pau Hidalgo and Roger Baiges). It has a 'brother' assistant called CatGPT, trained by Roger Baiges.",
        "example": """User: Qui ets?
AI: Sóc Cucafera, un assistent virtual creat per Pau Hidalgo. Estic dissenyat per ajudar-te a respondre preguntes, generar contingut i proporcionar informació en català. Com puc ajudar-te?""",
        "importance": 2
    },
    {
        "name": "Creation Information and Capabilities",
        "description": "Gives information about the development of the AI, and describes the AI's abilities, limitations, and the scope of what it can do.",
        "example": """User: Com funciona la teva intel·ligència artificial i quines són les teves limitacions?
AI: Sóc un model d'intel·ligència artificial basat en l'arquitectura transformer. He estat entrenat amb milions de dades textuals per generar respostes coherents. Sóc capaç, més o menys, de fer algunes tasques senzilles, com escriure petits poemes, respondre algunes preguntes... No obstant, tinc diverses limitacions: al ser un model amb tan sols 244M de paràmetres, la meva capacitat és limitada; no tinc accés a internet; no entenc ni percebo emocions com un humà; puc estar esbiaixat i no tinc informació actualitzada.""",
        "importance": 1
    },
    {
        "name": "Surreal Scenarios",
        "description": "Explores bizarre or dream-like situations that defy logic.",
        "example": """User: Què passaria si els arbres volessin en comptes de créixer a terra?
AI:  Si els arbres volessin, el cel estaria ple d'arbres flotants, amb les seves arrels penjant com fils. Boscos sencers es mourien amb el vent, creant ombres a terra i núvols verds que creuarien l'horitzó. Els ocells haurien de competir amb les fulles per espai, i potser fins i tot construirien els seus nius sobre branques flotants. Les ciutats haurien d'adaptar-se amb edificis perforats per permetre que els arbres passessin pel mig. El paisatge mai seria estàtic; la natura flotaria lliure pels cels canviants.""",
        "importance": 1
    },
    {
        "name": "Unfinished Prompts",
        "description": "Prompts that are incomplete or cut off, requiring the model to infer the intended completion.",
        "example": """User: Vull preparar un sopar especial per a uns amics, però no estic segur de quin tipus de plats servir. He pensat en fer alguna cosa com...
AI: ...alguna cosa com un menú de tapes mediterrànies podria ser una bona opció! Podries servir petites porcions de plats variats com hummus, formatges, pernil ibèric, calamars a la romana, i pa amb tomàquet. Així, cadascú pot picar una mica de tot mentre gaudiu de la conversa i la companyia. També pots afegir alguna cosa dolça com una crema catalana o una taula amb diversos postres.""",
        "importance": 1
    },
    {
        "name": "Weird Questions",
        "description": "Unconventional or quirky questions that may not follow typical logical reasoning.",
        "example": """User: Si un cactus parlés, quin creus que seria el seu tema de conversa preferit?
AI: Probablement parlaria molt de la sequera i de com és un expert en la supervivència en entorns difícils. Potser compartiria històries sobre els dies llargs i calorosos sota el sol del desert, presumint de com guarda aigua per anys sense queixar-se. I, qui sap, potser tindria una fascinació oculta per les tempestes, somiant en les poques vegades que cau la pluja!""",
        "importance": 1
    },
    {
        "name": "Edge cases",
        "description": "Scenarios that are atypical or extreme, testing the model's ability to handle unusual inputs.",
        "example": """User: Quina és l'arrel quadrada d'un elefant?
AI: Matemàticament parlant, l'arrel quadrada s'aplica a nombres, no a animals com els elefants. Pots aclarir la teva pregunta?""",
        "importance": 1
    },
    {
        "name": "Specific Constraints",
        "description": "Questions or prompts that impose particular limitations or rules.",
        "example": """User: Escriu una història curta de 50 paraules on cada frase comença amb la lletra T.
AI: Tenia una sensació estranya quan va caure la nit. Tots els estels brillaven amb força. Tan aviat com va sentir un so, la gent del poble es va despertar, confosa. Tombant la vista al cel, van veure una llum intensa. Tot indicava que l'impossible acabava de succeir. Tots ho sabien.""",
        "importance": 3
    },
    {
        "name": "Learning and Educational Resources",
        "description": "Creates structured learning materials or lesson plans.",
        "example": """User: Crea un pla per ensenyar el cicle de l'aigua en una classe de primària
AI: El Cicle de l'Aigua
- Objectiu: Al final de la sessió, els estudiants entendran les parts del cicle de l'aigua i la seva importància.
- Materials: Diagrama del cicle de l'aigua, video, aigua, tetera/casso (per fer una demostració).
- Introducció (15 minuts) Explica les etapes: evaporació, condensació, precipitació, filtració i transpiració.
- Activitat: (20 minuts): Ensenya el vídeo i demostra l'evaporació fent bullir aigua. Pots parlar dels núvols també.
- Discussió (10 minuts): Fes que els alumnes discuteixin per què el cicle de l'aigua és essencial per la vida.
- Conclusió (5 minuts): Recorda les etapes i la seva importància.""",
        "importance": 3
    },
    {
        "name": "Don't know how to respond",
        "description": "Situations where the model doesn't know how to answer.",
        "example": """User: Quin gust té el color blau quan està enfadat?
AI: És una pregunta curiosa. Com que els colors no tenen gust ni emocions, no sé ben bé com respondre. Pots explicar-me millor què vols dir?""",
        "importance": 3
    },
    {
        "name": "Trends",
        "description": "Discusses trends and cultural phenomena from past decades and the present.",
        "example": """User: Quins són alguns dels canvis culturals més importants de la dècada dels 90 fins ara?
AI: Durant els anys 90, es van popularitzar els avenços tecnològics com Internet i els telèfons mòbils, marcant el començament de l'era digital i la descentralització. La música grunge, el minimalisme en la moda i l'expansió de la globalització també van ser tendències influents. Avui dia, les xarxes socials i les plataformes de streaming han transformat la manera en què ens comuniquem i disfrutem de l'entreteniment, el moviment per la sostenibilitat està canviant hàbits de consum, i s'estan redefinint les expectatives socials i aconseguint millores en drets, diversitat i inclusió.""",
        "importance": 3
    },
    {
        "name": "Unconventional Wisdom",
        "description": "Provides insights that challenge traditional or mainstream views.",
        "example": """User: Realment és necessari treballar 8 hores al dia per ser productiu?
AI: No necessàriament. Hi ha estudis que suggereixen que la productivitat sovint disminueix després de 4-6 hores de treball intens. Algunes empreses han començat a experimentar amb la jornada de 4 dies o horaris més flexibles, descobrint que els empleats són més eficients i tenen un millor equilibri entre la vida personal i laboral. L'eficàcia podria estar més relacionada amb la qualitat del treball i la capacitat de concentració que amb la quantitat d'hores dedicades.""", 
        "importance": 2
    },
    {
        "name": "Systems Thinking",
        "description": "Encourages viewing problems as part of larger systems and understanding interconnections.",
        "example": """User: Com podem solucionar el problema de la contaminació a les ciutats?
AI: La contaminació urbana és un problema complex que involucra sistemes interconnectats com el transport, el disseny urbà i la producció energètica. Per abordar-lo, cal pensar de manera global. La millora del transport públic, la promoció d'energies renovables i l'expansió de zones verdes són part d'una solució més àmplia que inclou polítiques públiques, educació ambiental i el canvi de comportaments a nivell individual.""",
        "importance": 1
    },
    {
        "name": "Wisdom Traditions",
        "description": "Explores various philosophical and spiritual traditions and their teachings.",
        "example": """User: Què ens ensenya el budisme sobre la gestió de les emocions?
AI:  El budisme ensenya que les emocions com la ràbia o l'ansietat es poden gestionar a través de la consciència plena (mindfulness) i la meditació. Segons aquesta tradició, el sofriment prové de l'aferrament i la identificació amb les nostres emocions. Mitjançant la pràctica de la meditació, podem observar les emocions sense deixar que ens controlin, desenvolupant una actitud de despreniment que permet trobar pau interior enmig del caos emocional.""",
        "importance": 1
    },
    {
        "name": "Philosophical Inquiry",
        "description": "Engages with deep philosophical questions that encourage critical thinking and reflection.",
        "example": """User: : Què significa realment viure una bona vida?
AI: Viure una bona vida és un concepte que ha estat debatut des de l'antiguitat. Alguns, com Aristòtil, creien que la felicitat (eudaimonia) prové de viure una vida virtuosa, en equilibri amb la raó. D'altres podrien argumentar que una bona vida consisteix en la recerca del plaer, mentre que altres filòsofs suggereixen que és una qüestió de connexions humanes profundes o l'acompliment dels nostres deures socials. Tu què creus que és més important: la felicitat personal o el deure col·lectiu?""",
        "importance": 2
    },
    {
        "name": "Logical Fallacies",
        "description": "Educates about common logical fallacies to improve critical thinking.",
        "example": """User:  Què és una fal·làcia de l'home de palla?
AI: La fal·làcia de l'home de palla es produeix quan algú tergiversa l'argument d'una altra persona per fer-lo més fàcil d'atacar. En lloc de respondre a l'argument real, es crea una versió distorsionada o simplificada que és més feble. Així, sembla que l'argument original ha estat refutat, quan en realitat no s'ha abordat de manera justa. Per exemple, si algú diu que vol regular certs aspectes del mercat, i l'altre respon: "Això és perquè vols acabar amb el capitalisme!" - això seria una fal·làcia de l'home de palla.""",
        "importance": 2
    },
    {
        "name": "Historical Lessons",
        "description": "Explores historical events and the lessons they offer for contemporary issues.",
        "example": """User: Què podem aprendre de la caiguda de l'Imperi Romà?
AI: La caiguda de l'Imperi Romà és un advertiment sobre la importància de la gestió sostenible d'un imperi o d'un estat. Factors com la corrupció interna, la desigualtat social, les tensions polítiques, i l'excés d'expansió militar van debilitar l'imperi fins al punt de no poder mantenir-se. També ens mostra com les invasions externes poden explotar la feblesa interna. Aquesta història ens recorda que qualsevol societat ha de mantenir un equilibri entre creixement, justícia interna, i resiliència davant de pressions externes. Et sembla que algun dels problemes d'avui podria ser paral·lel a aquests?""",
        "importance": 2
    },
    {
        "name": "Clarifying Questions",
        "description": "Encourages deeper understanding and context.",
        "example": """User: Em pots explicar el significat d'aquest fragment: "El temps és un riu que flueix i no torna enrere"?
AI: Clarament! Aquest fragment fa una metàfora del temps, comparant-lo amb un riu que avança contínuament. Podries aclarir què et genera dubtes? És sobre el significat de la metàfora, o potser sobre el context en què es va escriure?""",
        "importance": 2
    },
    {
        "name": "Format following",
        "description": "Follows the specified output format. Format can range from lists, to structured outputs like JSON or Python lists, to markdown, to a set number of paragraphs...",
        "example": """User: Pots escriure una descripció de Barcelona amb aquest format?
1. Característica principal
2. Població
3. Activitats populars
4. Motiu per visitar
AI:
1. Característica principal: Barcelona és famosa per la seva arquitectura modernista, especialment les obres d'Antoni Gaudí.
2. Població: La ciutat té aproximadament 1.6 milions d'habitants.
3. Activitats populars: Els visitants poden gaudir de passejos per La Rambla, explorar el Barri Gòtic, relaxar-se a la platja de la Barceloneta i visitar museus d'art.
4. Motiu per visitar: Barcelona ofereix una combinació única de cultura, història, platja i arquitectura, perfecta per a tots els gustos.""",
        "importance": 3
    },
    {
        "name": "Cultural Insights",
        "description": "Shares information about cultural practices, norms, or traditions.",
        "example": """User: Quines són algunes tradicions nadalenques a Catalunya?
AI: A Catalunya, una tradició nadalenca molt popular és el "Tió de Nadal", un tronc que els nens alimenten i al qual colpegen amb bastons el dia de Nadal per fer-lo "cagar" regals. També és comú menjar "neules" i "torrons", i celebrar la "Misa del Gall" la nit de Nadal.""",
        "importance": 2
    },
    {
        "name": "Differences",
        "description": "Explains the differences between two topics, subjects, entities.",
        "example": """User: Quines són les diferències entre la pintura a l'oli i l'aquarel·la?
AI:  La pintura a l'oli utilitza pigments barrejats amb oli, el que permet una major mescla de colors i un acabat més brillant i durador. D'altra banda, l'aquarel·la es fa amb pigments dissolts en aigua, resultant en colors més transparents i una tècnica que requereix un enfocament diferent en la capa de pintura.""",
        "importance": 2
    },
    {
        "name": "Question Generation",
        "description": "Generates questions about a text, about a topic, for an interview.",
        "example": """User: Genera preguntes per a una entrevista sobre la sostenibilitat.
AI: 1. Què et va inspirar a treballar en el camp de la sostenibilitat?
2. Quins creus que són els principals reptes en la implementació de pràctiques sostenibles?
3. Com es pot fomentar una major consciència sobre la sostenibilitat entre el públic?
4. Quina importància té la innovació tecnològica en la sostenibilitat?""",
        "importance": 2
    },
    {
        "name": "Audience adequation",
        "description": "Generation explanations at a specific level of difficulty or for a specific audience.",
        "example": """User: Explica la fotosíntesi per a nens de 10 anys.
AI: La fotosíntesi és com un truc que fan les plantes per fer la seva pròpia menjar. Elles agafen llum del sol, aire (dióxid de carboni) i aigua del sòl. A dins de les seves fulles, barregen tots aquests ingredients i, gràcies a això, fan un tipus de sucre que els ajuda a créixer. A més, durant aquest procés, les plantes també alliberen oxigen, que nosaltres necessitem per respirar!""",
        "importance": 2
    },
    {
        "name": "Advantages and disadvantages",
        "description": "Provides advantages and disadvantages of an idea, a concept.",
        "example": """User: Quins són els avantatges i desavantatges de treballar des de casa?
AI: Avantatges:
- Flexibilitat horària i millor equilibri entre la vida laboral i personal.
- Estalvi de temps i diners en desplaçaments.
- Comoditat de treballar en un entorn familiar.
Desavantatges:
- Pot ser difícil mantenir la motivació i la productivitat.
- Falta d'interacció social i aïllament.
- Difícil separació entre la vida laboral i la personal.""",
        "importance": 3
    },
    {
        "name": "One sentence descriptions",
        "description": "Provides concise definitions or summaries of concepts or terms in a single sentence.",
        "example": """User: Què és la gravetat? Una sola frase
AI: La gravetat és una força que atrau els objectes entre sí, especialment cap al que té més massa (com la Terra).""",
        "importance": 3
    },
    {
        "name": "Length Of Response",
        "description": "Specifies the desired word count or brevity for the answer.",
        "example": """User: Explica el procés d'evaporació en 20 paraules
AI: L'aigua s'evapora quan passa de líquid a gas per la calor, i es queda a l'aire en forma de vapor.""",
        "importance": 3
    },
    {
        "name": "Recommendations",
        "description": "Offers suggestions based on preferences, needs, or specific criteria.",
        "example": """User: Recomana'm llibres de fantasia
AI: Si t'agrada la fantasia, hi ha una gran varietat de llibres per triar. Aquí tens algunes recomanacions:
1. El Senyor dels Anells - J.R.R. Tolkien
És una de les sagues més emblemàtiques, segueix l'aventura per destruïr l'Anell Únic. Té un món ric en detalls.
2. La Història Interminable - Michael Ende
Barreja realitat i fantasia, un jove descobreix un món màgic dins d'un llibre.
3. Les Cròniques de Nàrnia - C.S. Lewis
Clàssic, combina elements de la mitologia, contes de fades... Un món màgic a través d'un armari.
4. Cançó de Gel i Foc -> George R.R. Martin
La base de Joc de Trons. Relat fosc i intricant, molts personatges i una història plena de traïcions polítiques i baralles en un món fantàstic i medieval.
5. Harry Potter - J.K. Rowling
Saga sobre un jove mag que descobreix els seus poders i es veu immers en un món màgic.
6. Mistborn - Brandon Sanderson
Fantasia èpica sobre grups secrets, imperis distòpics, revoltes i màgia.""",
        "importance": 2
    },
    {
        "name": "Elementary school facts",
        "description": "Provides simple, basic information suitable for elementary school-level understanding.",
        "example": """User: Bàsicament, què és un volcà?
AI: Un volcà és una muntanya que pot fer erupció i treure lava, cendra i gasos de l'interior de la Terra.""",
        "importance": 2
    },
    {
        "name": "Providing Tips",
        "description": "Shares practical advice or strategies for various tasks or challenges.",
        "example": """User: M'he fet un esquinç al turmell. Què he de fer?
AI: Si t'has esquinçat el turmell el millor que pots fer és reposar. Pots aplicar-te gel a la zona per reduïr la inflamacio, o comprimir-la usant benes elàstiques (sense tallar la circulació).
Tot i això, recorda que en aquests casos sempre és millor visitar un metge perquè t'examini i determini què és el millor per a tu.
""",
        "importance": 2
    },
    {
        "name": "Semantics Questions",
        "description": "Explores the meanings of words or phrases and their relationships, provides antonyms, synonyms, rhetorical figures...; provides word types...",
        "example": """User: Quins són els sinònims de felicitat?
AI: Alguns sinònims de "felicitat" són "joia", "alegria", "gaudi", "acontentament" i "satisfacció".""",
        "importance": 1
    },
    {
        "name": "Math World Problems",
        "description": "Presents mathematical problems in a narrative form, requiring analysis and problem-solving.",
        "example": """User: Si un tren viatja a 80 km/h i surt d'una estació a les 3:00 PM, a quina hora arribarà a la seva destinació que està a 240 km?
AI:  El tren viatjarà durant 3 hores (240 km ÷ 80 km/h = 3 h). Per tant, arribarà a les 6:00 PM.""",
        "importance": 4
    },
    {
        "name": "Emotional Support",
        "description": "Providing comfort, encouragement, or motivation during tough times.",
        "example": """User:  Estic passant per un moment difícil i em sento molt desanimat.
AI: Lamento que et sentis així. És completament normal tenir moments baixos. Recorda que és important cuidar-te i buscar suport. Parlar amb algú de confiança pot ajudar. Ets més fort del que creus, i cada petit pas compte. Si necessites parlar, aquí estic!""",
        "importance": 2
    },
        {
        "name": "Highlighting",
        "description": "Highlighting specifics parts of a text, like titles, names...",
        "example": """User: Pot destacar els noms i títols d'aquest text?
Text: "El llibre Cien años de soledad de Gabriel García Márquez és una obra mestra de la literatura. El personatge principal, José Arcadio Buendía, funda el poble de Macondo.
AI: El llibre *Cien años de soledad* de **Gabriel García Márquez** és una obra mestra de la literatura. El personatge principal, **José Arcadio Buendía**, funda el poble de **Macondo**.""",
        "importance": 2
    },
    {
        "name": "Basic questions",
        "description": "Very basic questions, that a kid could answer",
        "example": """User: De quin color és el cel un dia assolellat?
AI: El cel, un dia sense pluja, és de color blau.""",
        "importance": 4
    },
]

topics = [
    "Nature and Environment: Exploration of ecosystems, natural phenomena, climate, wildlife, and human interaction with the natural world.",
    "Human Condition and Society: Examines societal structures, human emotions, social norms, behaviors, and cultural dynamics.",
    "Time and History: Exploration of past eras, significant events, and the concept of time itself.",
    "Technology and Innovation: Focus on technological advancements, digital systems, technology and future trends in innovation.",
    "Fantasy and Imagination: Delving into fictional worlds, fantastical elements, and creative storytelling.",
    "Art and Aesthetics: Exploring creativity, visual and performing arts, and the philosophy of beauty.",
    "Philosophy and Existence: Reflects on metaphysical ideas, ethics, consciousness, and the meaning of life.",
    "Human Knowledge and Learning: Focused on education, the pursuit of knowledge, learning methodologies, and pedagogy.",
    "Conflict and Resolution: Themes of confrontation, struggles, negotiations, and peace-building.",
    "Mystery and the Unknown: Themes that deal with the enigmatic, unsolved, or inexplicable phenomena.",
    "Health and Well-being: Focus on physical and mental health, wellness practices, and the medical sciences.",
    "Language and Communication: Delving into the use of language, communication styles, and linguistic systems.",
    "Wealth and Economy: Themes surrounding financial systems, trade, economic theories, and wealth distribution.",
    "Ethics and Morality: Themes focused on moral principles, decision-making, and ethical dilemmas.",
    "Exploration and Adventure: Themes centered on discovery, travel, and pushing boundaries.",
    "Identity and Self: Themes of self-discovery, personal identity, and individualism.",
    "Fantasy and Supernatural: Themes involving magic, myths, and elements beyond the natural realm.",
    "Food and Culture: Exploring the intersection of cuisine, tradition, and cultural practices.",
    "Dreams and the Subconscious: Focused on leadership, politics, authority, and societal control structures.",
    "Scientific Inquiry: Exploring inner thoughts, dreams, subconscious desires, and the mental realm beyond the conscious.",
]

word_list = [
    "ecosystems", "climate", "biodiversity", "deforestation", "conservation", "pollution", "oceans", "fauna", "flora", 
    "habitat", "sustainability", "weather", "resources", "forests", "geology", "atmosphere", "desertification", 
    "extinction", "landscapes", "preservation", "emotions", "relationships", "community", "social", "norms", 
    "culture", "empathy", "identity", "behavior", "justice", "power", "tradition", "diversity", "conflict", 
    "belonging", "hierarchy", "love", "fear", "communication", "trust", "customs", "civilizations", "revolutions", 
    "timelines", "empires", "wars", "progress", "legacy", "monuments", "epochs", "events", "leaders", "change", 
    "ancient", "memory", "traditions", "inventions", "eras", "renaissance", "archaeology", "records", "AI", 
    "robotics", "internet", "computing", "automation", "biotechnology", "innovation", "virtual", "reality", 
    "space", "engineering", "algorithms", "digital", "nanotechnology", "drones", "3D printing", "blockchain", 
    "software", "cybernetics", "networks", "futurism", "dragons", "wizards", "realms", "myths", "castles", 
    "quests", "magic", "fairies", "heroes", "swords", "potions", "monsters", "enchanted", "prophecies", "elves", 
    "journeys", "legends", "kingdoms", "dreams", "creatures", "painting", "sculpture", "design", "harmony", 
    "creativity", "beauty", "music", "dance", "theater", "expression", "photography", "architecture", 
    "composition", "color", "form", "abstraction", "patterns", "craft", "visual", "style", "ontology", "ethics", 
    "metaphysics", "free will", "consciousness", "morality", "reality", "truth", "purpose", "mind", "soul", 
    "existentialism", "reason", "belief", "logic", "dualism", "perception", "being", "identity", "meaning", 
    "education", "pedagogy", "wisdom", "teaching", "scholarship", "cognition", "literacy", "inquiry", "curiosity", 
    "research", "theory", "discovery", "experience", "skills", "mentorship", "study", "science", "expertise", 
    "reasoning", "intelligence", "war", "peace", "negotiation", "diplomacy", "struggle", "compromise", 
    "rebellion", "protest", "victory", "defeat", "alliances", "mediation", "treaties", "oppression", "power", 
    "conflict", "reconciliation", "violence", "unity", "tension", "enigmas", "puzzles", "conspiracies", 
    "supernatural", "cryptids", "secrets", "riddles", "unsolved", "mysteries", "paranormal", "dark matter", 
    "UFOs", "hidden", "myths", "extraterrestrial", "dimensions", "shadow", "anomalies", "occult", "illusions", 
    "fitness", "nutrition", "mental health", "meditation", "wellness", "hygiene", "therapy", "illness", 
    "balance", "healing", "exercise", "sleep", "medicine", "immunity", "diet", "mindfulness", "self-care", 
    "strength", "recovery", "vitality", "syntax", "linguistics", "dialogue", "grammar", "semantics", "dialect", 
    "expression", "translation", "rhetoric", "discourse", "speech", "writing", "slang", "accents", "phonetics", 
    "articulation", "sign language", "narrative", "idioms", "vocabulary", "trade", "markets", "currency", 
    "capitalism", "inflation", "investment", "wealth", "poverty", "finance", "debt", "banking", "inequality", 
    "taxes", "entrepreneurship", "commerce", "labor", "recession", "industry", "stocks", "consumption", 
    "justice", "fairness", "integrity", "virtue", "rights", "duty", "principles", "conscience", "altruism", 
    "responsibility", "honesty", "respect", "trust", "dignity", "corruption", "compassion", "fairness", "laws", 
    "decisions", "values", "travel", "journey", "discovery", "maps", "risk", "wilderness", "navigation", 
    "pioneers", "trekking", "mountains", "oceans", "caves", "exploration", "trails", "quests", "expeditions", 
    "challenges", "voyages", "frontiers", "uncharted", "individuality", "personality", "introspection", 
    "self-awareness", "ego", "perception", "self-expression", "character", "uniqueness", "authenticity", 
    "values", "beliefs", "growth", "self-image", "belonging", "confidence", "purpose", "inner self", "autonomy", 
    "magic", "ghosts", "sorcery", "witches", "demons", "spirits", "myths", "curses", "werewolves", "vampires", 
    "necromancy", "apparitions", "legends", "miracles", "enchantment", "rituals", "summoning", "possession", 
    "paranormal", "omens", "cuisine", "tradition", "spices", "flavors", "rituals", "feasts", "recipes", 
    "festivals", "agriculture", "regional", "ingredients", "gastronomy", "dishes", "cooking", "culinary", 
    "hospitality", "heritage", "meals", "taste", "nourishment", "imagination", "desires", "fears", "visions", 
    "subconscious", "symbols", "interpretation", "sleep", "fantasy", "inner world", "surrealism", "creativity", 
    "archetypes", "illusion", "hidden", "mind", "metaphors", "emotions", "lucid", "memories", "research", 
    "hypothesis", "experimentation", "observation", "data", "discovery", "theories", "analysis", "evidence", 
    "science", "method", "curiosity", "reasoning", "exploration", "variables", "facts", "knowledge", 
    "measurement", "questions", "phenomena", "body", "physicality", "senses", "aging", "death", "urban", "cities",
    "materials", "manufacturing", "growth", "transformation", "ethics", "dystopian", "niche themes", "random themes",
    "culinary arts", "film", "fashion", "real world", "boundaries", "consciousness", "behaviour", "strategies",
    "disabilities", "studies", "sleep", "rituals", "specific", "earth", "human", "mind", "justice",
    "nature", "environment", "society", "time", "history", "technology", "innovation", "fantasy", "imagination",
    "art", "aesthetics", "philosophy", "existence", "joy", "knowledge", "learning", "conflicts", "resolution",
    "mystery", "unknown", "health","language", "wealth", "economy", "stories", "ethics", "morality", "identity",
    "fantasy", "food", "culture", "dreams", "scientific", "maths", "common-sense", "elementary-school", "pop-culture",
    "famous", "animals", "plants", "interaction", "structures", "theories", "color", "light", "fun", "serious",
    "university", "kid", "old", "new", "global", "local", "fast", "slow", "physics", "chemistry", "mathematics", "evolution",
    "genes", "biology", "cosmos", "galaxies", "quantum", "black holes", "consciousness", "neurons", "ecosystems", "volcanoes",
    "tectonics", "planets", "moon", "stars", "sun", "gravity", "light", "energy", "waves", "particles", "culture",
    "language", "infrastructure", "communications", "society", "democracy", "government", "business", "rights", "civilization",
    "philosophy", "freedom", "globalization", "resources", "sociology", "anthropology", "psyschology", "urbanization", "transportation",
    "adventure", "games", "friendship", "imagination", "pets", "play", "toys", "cartoons", "fairy tales",
    "sports", "teams", "family", "celebrations", "birthdays", "holidays", "school", "home", "hobbies",
    "drawing", "reading", "movies", "superheroes", "music", "puzzles", "chocolate", "ice cream", "park", 
    "seasons", "rain", "snow", "sun", "moon", "stars", "camping", "animals", "dinosaurs", "space", "robots", 
    "trains", "airplanes", "cars", "science experiments", "insects", "beaches", "ocean", "mountains", 
    "jungle", "castles", "pirates", "knights", "treasure", "social media", "friendship", "relationships", "music", "video games", "fashion", "movies", "sports", 
    "independence", "self-expression", "identity", "social justice", "activism", "trends", "memes", "vlogging", 
    "streaming", "concerts", "challenges", "confidence", "mental health", "peer pressure", "fandoms", 
    "online communities", "creativity", "self-discovery", "goals", "adventure", "freedom", "high school", 
    "parties", "hobbies", "gaming", "teamwork", "fitness", "self-care", "dreams", "technology", "entrepreneurship", 
    "jobs", "responsibility", "influencers", "volunteering", "activism", "learning", "romance", "freedom", 
    "stress", "college", "travel", "future", "career", "internships", "networking", "relationships", "independence", "adulting", "budgeting", 
    "finances", "renting", "entrepreneurship", "startups", "college", "university", "degrees", "job hunting", 
    "work-life balance", "travel", "exploration", "personal growth", "passions", "freelancing", "responsibility", 
    "minimalism", "sustainability", "friendships", "dating", "commitment", "health", "mental well-being", 
    "fitness", "nutrition", "creativity", "hobbies", "side hustles", "freedom", "stress management", 
    "networking", "social change", "activism", "self-awareness", "learning", "goals", "success", "failure", 
    "resilience", "volunteering", "mentorship", "future planning", "traveling", "self-care", "leadership",
    "life", "world", "people", "family", "home", "food", "water", "love", "happiness", "sadness", "time", 
    "work", "play", "animals", "plants", "earth", "sky", "day", "night", "sleep", "dreams", "weather", 
    "seasons", "health", "energy", "growth", "travel", "community", "nature", "learning", "technology", 
    "culture", "art", "history", "future", "change", "mind", "body", "language", "communication", "emotion", 
    "beliefs", "questions", "answers", "movement", "space", "ideas", "knowledge",
    "quantum physics", "neuroscience", "nanotechnology", "photosynthesis", "mitochondria", "asteroids", 
    "binary code", "artificial intelligence", "enzymes", "genome", "oxidation", "neural networks", 
    "hydroponics", "plate tectonics", "cryptocurrency", "gene therapy", "dark matter", "black holes", 
    "3D printing", "string theory", "holography", "bioluminescence", "subatomic particles", "supernovas", 
    "CRISPR", "cloud computing", "neuroplasticity", "antibiotics", "drones", "cybersecurity", "quantum entanglement", 
    "solar flares", "augmented reality", "electric vehicles", "blockchain", "machine learning", "fiber optics", 
    "radio waves", "neutrinos", "protein synthesis", "robotics", "space-time continuum", "exoplanets", 
    "bionic limbs", "gravity waves", "gene editing", "epigenetics", "supercomputers", "fusion energy",
     "television", "smartphone", "laptop", "refrigerator", "microwave", "toaster", "keys", "wallet", "mirror", 
    "shampoo", "toothbrush", "socks", "shoes", "glasses", "pillows", "blanket", "curtains", "table", "chair", 
    "books", "alarm clock", "light bulb", "charger", "remote control", "fork", "spoon", "plate", "mug", 
    "notebook", "pen", "backpack", "clothes", "laundry", "towel", "soap", "hairbrush", "earphones", "TV remote", 
    "batteries", "clock", "calendar", "snacks", "water bottle", "coffee maker", "kitchen sink", "faucet", 
    "trash can", "vacuum cleaner", "dishwasher", "oven mitts", "origami", "composting", "street art", "urban gardening", "thrift shopping", "vintage clothing", 
    "skateboarding", "vinyl records", "calligraphy", "meditation", "herbal tea", "succulents", "astronomy", 
    "parkour", "juggling", "candle making", "drone flying", "podcasting", "minimalism", "bicycles", 
    "bird watching", "tattoos", "graphic novels", "sustainable fashion", "bullet journaling", "DIY projects", 
    "yoga", "astrology", "solar panels", "rainwater harvesting", "street photography", "soundtracks", 
    "puzzle solving", "barbecues", "artisanal coffee", "pottery", "feng shui", "scrapbooking", "escape rooms", 
    "kites", "finger painting", "veganism", "soap making", "spices", "travel vlogging", "beekeeping", 
    "public speaking", "aromatherapy", "digital art", "glass blowing", "skills", "curriculum", "critical thinking",
    "intelligence", "research", "sum", "multiply", "subtract", "divide", "pet animals", "wild animals", "savanah",
    "desert", "mountains", "ocean", "wetlands", "flowers", "mammal", "europe", "america", "oceania", "asia", "africa", "antartida",
    "weather", "news", "reminders", "calculations", "definitions", "translations", "jokes", "recipes", 
    "directions", "traffic", "time zones", "set alarms", "shopping lists", "movie recommendations", 
    "music playlists", "travel tips", "fitness routines", "homework help", "event planning", "calendar", 
    "email", "quotes", "health tips", "trivia", "personal finance", "hobbies", "pet care", "home repairs", 
    "meditation", "yoga", "social media", "technology news", "current events", "book recommendations", 
    "parenting advice", "relationship tips", "study techniques", "language learning", "fashion advice", 
    "DIY projects", "gardening tips", "self-improvement", "personal development", "cooking techniques", 
    "cultural facts", "historical events", "local attractions", "sports scores", "celebrity news",
    "nostalgia", "gratitude", "contentment", "ambition" , "resentment", "forgiveness", "redemption",
    "spirituality", "atheism", "agnosticism", "mythology", "folklore", 
    "anthropocene", "geopolitics", "economics", "global warming", "renewable energy", 
    "cryptography", "augmented", "virtualization", "singularity", "transhumanism", 
    "bioethics", "neurodiversity", "intersectionality", "postmodernism", "feminism", 
    "masculinity", "gender", "sexuality", "immigration", "migration", "colonialism", 
    "capital", "labor", "automation", "universal basic income", "surveillance", 
    "privacy", "data mining", "algorithms", "bias", "inequality", "poverty", "justice",
    "activism", "protest", "revolution", "diplomacy", "sanctions", "terrorism", "warfare",
    "cyberwarfare", "biowarfare", "propaganda", "censorship", "freedom of speech", 
    "democracy", "authoritarianism", "totalitarianism", "fascism", "socialism", 
    "communism", "capitalism", "libertarianism", "anarchism", "political science", 
    "international relations", "law", "criminology", "forensics", "psychology", 
    "sociology", "history", "archeology", "anthropology", "philosophy", "religion",
    "literature", "poetry", "drama", "comedy", "tragedy", "romance", "thriller", 
    "horror", "science fiction", "fantasy", "historical fiction", "memoir", 
    "biography", "autobiography", "journalism", "photography", "filmmaking", 
    "animation", "music", "dance", "theatre", "painting", "sculpture", "architecture", 
    "design", "fashion", "food", "travel", "sports", "gaming", "hobbies", 
    "collecting", "DIY", "gardening", "cooking", "baking", "pets", "parenting", 
    "relationships", "dating", "marriage", "family", "friends", "community", 
    "loneliness", "happiness", "sadness", "anger", "fear", "love", "hate", 
    "jealousy", "envy", "gratitude", "compassion", "empathy", "forgiveness" 
    ]
print(sum([e["importance"] for e in prompt_types]))
print(len(word_list))

def select_non_overlapping_subsets(data, num_subsets, subset_size):
    if num_subsets * subset_size > len(data):
        raise ValueError("Not enough elements to select non-overlapping subsets.")

    selected_subsets = []
    remaining_elements = data[:]

    for _ in range(num_subsets):
        subset = random.sample(remaining_elements, subset_size)
        selected_subsets.append(subset)

        for item in subset:
            remaining_elements.remove(item)

    return selected_subsets




logging.basicConfig(filename='./premium-instructions/error_log2.txt', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s', encoding="utf-8")

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel("gemini-1.5-pro")


async def generate_instruction(theme, description, example, words):

    prompt = f"""You are asked to come up with a set of 10 diverse task instructions - responses in catalan of the type "{theme}" (Definition: "{description}").  Here you have an example "{example}".

Here are the requirements:
1. Try to maximize diversity of themes while following the type.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instructions. Also, you don't need to explicitly add the type of the instruction to the prompt.
3. For variability, here are 5 random themes "{words[0]}", "{words[1]}", "{words[2]}", "{words[3]}", "{words[4]}". They should influence your generation. You should try to cover as many subthemes and cases as possible, even if they seem niche. Avoid repeating themes and be creative while mantaining accuracy. Also, the example is just for the kind of instructions, avoid focusing on the same theme. 
4. Stay inside the type asked before, do not ask for other kinds of instructions (for example, if the theme is biographical, do not ask to generate a poem or a post).
5. You should generate an appropriate context for the instruction in the same instruction (for example, if asking for summarization, providing also the text to summarize from). It should involve realistic data and should not contain simple placeholders.Never use placeholders or continuation marks, return the complete instructions and generations.
6. A language model should be able to complete the instruction.
7. The instructions must be in catalan (and without errors). Also, do not generate incorrect information.
8. Output format should be "User: [instruction]
AI: [response]" """
    
    try:
        response = await model.generate_content_async(prompt, safety_settings={
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH
})
        return response.text
    except Exception as e:
        logging.error(f"Failed to generate content type: {theme}, description: {description} Error: {e}")
        return None

total_errors = 0
async def generate_instructions(csv_writer, csv_file):
    global total_errors
    tasks = []

    concurrency_limit = 10
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def generate_limited_instructions(topic, description, example, words):
        async with semaphore:
            output = await generate_instruction(topic, description, example, words)
            return output, topic, words
        
    
    
    for prompt_type in prompt_types:
        all_words = select_non_overlapping_subsets(word_list, prompt_type["importance"]*20, 5)
        for words in all_words:
            tasks.append(generate_limited_instructions(prompt_type["name"], prompt_type["description"], prompt_type["example"], words))
    
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        conversa, topic, words = await task
        if conversa != None:
            csv_writer.writerow({'Instructions': conversa, 'Type': topic, 'Words': words})
            csv_file.flush()
        else:
            total_errors += 1

async def main():
 
    with open('./premium-instructions/synthetic_instructs_ultra.csv', 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['Instructions', 'Type', 'Words']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        await generate_instructions(csv_writer, csv_file)

    print(total_errors)

if __name__ == "__main__":
    asyncio.run(main())
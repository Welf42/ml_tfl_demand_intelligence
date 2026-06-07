# Data

## Source

**Transport for London Journey Information**  
Kaggle: <https://www.kaggle.com/datasets/astronasko/transport-for-london-journey-information>  
Original: <https://data.london.gov.uk/dataset/oyster-card-journey-information/>

Anonymized Oyster card journey records exported from TfL's ticketing system. The file `Nov09JnyExport.csv.zip` covers journeys from November 2009.

## Schema

| Column | Type | Description |
|---|---|---|
| `downo` | int | Day-of-week number |
| `daytype` | str | Day type (Mon, Tue, …, Sat, Sun) |
| `SubSystem` | str | Transport mode — LUL (Tube), LRC (Rail), LBL (Bus), DLR, etc. |
| `StartStn` | str | Origin station or stop |
| `EndStation` | str | Destination station or stop |
| `EntTime` | int | Entry time in minutes from midnight |
| `EntTimeHHMM` | str | Entry time as HH:MM |
| `ExTime` | int | Exit time in minutes from midnight |
| `EXTimeHHMM` | str | Exit time as HH:MM |
| `ZVPPT` | str | Zone-pair code (origin–destination fare zones) |
| `JNYTYP` | str | Journey type (TKT = ticket gate, etc.) |
| `DailyCapping` | str | Daily capping applied (Y/N) |
| `FFare` | int | Full fare (pence) |
| `DFare` | int | Discounted fare (pence) |
| `RouteID` | str | Route identifier (XX = not applicable) |
| `FinalProduct` | str | Fare product used (Travelcard, Freedom Pass, etc.) |

## Notes

- `StartStn` is `"Unstarted"` when the tap-in was not recorded (common for bus or incomplete journeys).
- Fares are integers in pence; 0 indicates free or capped travel.
- Zone-pair codes (`ZVPPT`) can proxy OD distance for journeys missing coordinates.

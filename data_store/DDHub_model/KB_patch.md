# Workflow

The workflow to generate the final TTL that serves as KB is as follows:

1. The TTL file from [DWIS project](https://github.com/D-WIS/DDHub-Semantic-Interoperability/blob/main/docs/vocabulary_development/auto-generated/rdf/DWISVocabulary.ttl) as base model.
2. Read all the patch knowledge in Excel files, and convert them to SPARQL.
3. Apply the the SPARQL to the base model.
4. Merge the TTL files from some temporary sources.

Thus, the new patch can be put in:

1. The Excel files: For those related to PrototypeData, Unit, Quantity. But the functions to process have to be adapted accordingly.
2. patch_create.ttl: If the patch can not be represented in Excel, manually represent them here.
3. patch_delete_update.ttl: If some deletion and update need to be made, manually represent them here.

# Supplement

## Unit

ThousandKilogramForce, or KiloKilogramForce, kkgf
ppm
Percent is also dimensionless.
TonForce: for ton in the context of hookload

## PrototypeData

(Human can interpret, but it is not in KB)

Tank volume (active)
Iso-butane (IC4), gas concentration?
Bit on Bottom, boolean?
On Bottom Status
true vertical depth
MAG_DEC
GTF_RT : MWD Gravity Toolface
GS_CHKP : Casing (choke) pressure
BITRUN : Bit run number
FVOC : Fill/gain volume obs. (cum)
SHKTOT_RT : MWD Total Shocks
STWT : String weight (rot,avg)
MWD_POS : MWD Frame Position
GRM1 : MWD Gamma Ray (API BH corrected)
MWD_VAL : MWD Transmitted Counts
TCHR : Time chrom sample
ROPI : Inverse ROP
GS_GASA : Gas (avg)
TRPOUT_CT : zzz:undefined
FRIC : Rotating friction factor
RSUX : Running speed-up (max)
GRR : MWD Raw Gamma Ray
MWD_TS : MWD Time Stamp
TCNF : MWD TF Bit Confidence Flag
MWD_SYNC : MWD Sync Status
MWD_ID : MWD DATP ID
PMPT : Pump Time
SHKRSK_RT : MWD Shock Risk
DCHV : Chromatograph Sample (TVD)
SHKPK_RT : MWD Shock Peak
MWD_DEL : MWD Delay Time
PASS_NAME : Pass Name
BDTI : Bit Drill Time
DRGF : DRAG sliding friction factor
MTF_RT : MWD Magnetic Toolface
TQLS : Torque loss
DCHR : Date of Chromatograph Sample
SHKRATE_RT : PowerUP Shock Rate
MWD_FRM : MWD Frame ID
GS_DCHM : Depth chrom sample (meas)
TRPM_RT : MWD Turbine RPM
TSTK : Total Strokes
TREV : Total Bit Revolutions
STIS : Slips stat (1=Out,0=In)
SRVTYPE
TIME : Time
GS_TVCA : Tank volume change (active)

...
Are they defined properly?
SPM, PumpRate, MPDPumpRate, RPM are often mistaken.

## Quantity

Electric voltage?
GravitationalLoadQuantity is MassQuantity
Gamma Ray measurements in drilling are typically in API units, counts per second (CPS), or similar—not frequency (1/s).
DPI, DotPerInch, counts per Length

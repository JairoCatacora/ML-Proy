# Datos

Los archivos crudos se descargan desde la fuente oficial de NYC Taxi & Limousine Commission (TLC).

## Archivos utilizados en la entrega previa

- `yellow_tripdata_2026-01.parquet`
- `yellow_tripdata_2026-02.parquet`
- `yellow_tripdata_2026-03.parquet`

## URLs directas

```text
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-01.parquet
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-02.parquet
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-03.parquet
```

Página oficial:

https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Diccionario de datos:

https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf

## Por qué no se versionan los Parquet

Cada archivo mensual contiene millones de registros. Para mantener el repositorio liviano y reproducible, `data/*.parquet` está excluido mediante `.gitignore`.

El notebook descarga los archivos automáticamente si no existen.

## Si la descarga automática falla

Descargar manualmente los tres archivos desde la página oficial y colocarlos dentro de:

```text
data/
```

Luego volver a ejecutar el notebook.

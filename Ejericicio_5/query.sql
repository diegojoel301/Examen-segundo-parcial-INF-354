CREATE PROCEDURE CompararPalabras
    @palabra1 varchar(20),
    @palabra2 varchar(20)
AS
BEGIN
    DECLARE @longitud1 int,
            @longitud2 int,
            @posicion int,
            @letra varchar(2),
            @contador int,
            @sql nvarchar(2000),
            @columna varchar(4),
            @contar int

    SET @longitud1 = LEN(@palabra1)
    SET @longitud2 = LEN(@palabra2)

    SELECT @posicion = 1
    SELECT @sql = 'CREATE TABLE nombre ('
    WHILE @posicion <= @longitud1
    BEGIN
        SELECT @letra = LEFT(@palabra1, 1)
        SELECT @palabra1 = RIGHT(@palabra1, LEN(@palabra1) - 1)
        SELECT @sql = @sql + @letra + CAST(@posicion AS varchar) + ' int, '
        SELECT @posicion = @posicion + 1
    END
    SELECT @sql = LEFT(@sql, LEN(@sql) - 1)
    SELECT @sql = @sql + ')'
    EXEC sp_executesql @sql

    SELECT @posicion = 1
    WHILE @posicion <= @longitud2
    BEGIN
        SELECT @letra = LEFT(@palabra2, 1)
        SELECT @palabra2 = RIGHT(@palabra2, LEN(@palabra2) - 1)
        SELECT @contar = COUNT(*)
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'nombre'
        AND LEFT(COLUMN_NAME, 1) = @letra
        AND ORDINAL_POSITION <= @posicion
        IF @contar > 0
        BEGIN
            SELECT TOP 1 @columna = COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = 'nombre'
            AND LEFT(COLUMN_NAME, 1) = @letra
            AND ORDINAL_POSITION >= @posicion
            SELECT @sql = 'INSERT INTO nombre(' + @columna + ') VALUES(1)'
            EXEC sp_executesql @sql
        END
        SELECT @posicion = @posicion + 1
    END

    SET @sql = 'SELECT '
    SELECT @contar = COUNT(*)
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'nombre'
    SELECT @posicion = 1
    WHILE @posicion <= @contar
    BEGIN
        SELECT @columna = COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = 'nombre' AND ORDINAL_POSITION = @posicion
        SET @sql = @sql + 'SUM(ISNULL(' + @columna + ',0)) +'
        SELECT @posicion = @posicion + 1
    END
    SELECT @sql = LEFT(@sql, LEN(@sql) - 1) + ' FROM nombre'
    EXEC sp_executesql @sql

    -- DROP TABLE nombre
END


DROP table nombre;

EXEC CompararPalabras @palabra1 = 'martha', @palabra2 = 'maria';

import requests
from datetime import datetime,timedelta
import sys
from lxml import html

## Utility to get the weather forecast from 'www.tutiempo.net'

min_date_in = sys.argv[1] # min date to export in format yyyy-mm-dd
max_date_in = sys.argv[2] # max date to export in format yyyy-mm-dd
csv_file_name = sys.argv[3] # output file
st_name_in = sys.argv[4] # weather station ICAO

## ICAO examples
## Malaga -> 'lemg'
## Granada -> legr
## Sevilla -> lezl
## Cordoba -> leba
## Cadiz, Jerez -> lejr
## Almeria, Huercal Overa -> leam
## Granada, Armilla -> lega

min_date = datetime.strptime(min_date_in,'%Y-%m-%d')
max_date = datetime.strptime(max_date_in,'%Y-%m-%d')

if not st_name_in:
    exit()

csv_file = open(csv_file_name, "w")

tmp_date = min_date
headers = {'Host':'www.tutiempo.net','user-agent':'Mozilla/5.0', 'accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8','Accept-Language':'es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3','Referer':'http://www.tutiempo.net/registros/lemg','DNT':'1','Upgrade-Insecure-Requests':'1' }

fst = 'https://www.tutiempo.net/registros/' + st_name_in

while tmp_date <= max_date:
    print tmp_date.strftime('%Y-%m-%d')
    r = requests.post(fst, data = {'date':tmp_date.strftime('%d-%m-%Y')}, headers=headers)    
    tree = html.fromstring(r.content)
    div = tree.get_element_by_id('HistoricosData')
    trs = div.xpath('.//tr')
    for tr in trs:
        tds = tr.xpath('td')
        # 0-Hora;2-Condiciones meteorologicas;3-Temperatura;5-Viento;6-Humedad;7-Presion atmosferica
        # 0-Hora;1-Estado;2-Temperatura;3-Viento;4-Humedad;5-Presion atmosferica
        hora = 'NA'
        if len(tds)==8:
            hora = tds[0].text_content().encode('utf-8')
            #estado = tds[1].xpath('div/@title')[0]
            estado = tds[2].text_content().encode('utf-8')
            temperatura = tds[3].text_content().encode('utf-8').replace('\xc2\xb0C','')
            viento = tds[5].text_content().encode('utf-8')
            humedad = tds[6].text_content().encode('utf-8')
            presion = tds[7].text_content().encode('utf-8')
        elif len(tds)==6:
            hora = tds[0].text_content().encode('utf-8')
            estado = tds[1].text_content().encode('utf-8')
            temperatura = tds[2].text_content().encode('utf-8').replace('\xc2\xb0C','')
            viento = tds[3].text_content().encode('utf-8')
            humedad = tds[4].text_content().encode('utf-8')
            presion = tds[5].text_content().encode('utf-8')
        if not hora=='NA':
            csv_file.write( tmp_date.strftime('%Y-%m-%d') + '|' + hora + '|' + estado + '|' + temperatura + '|' + viento + '|' + humedad + '|' + presion + '\n')
    # end for
    tmp_date = tmp_date + timedelta(days=1)
# end while

csv_file.close()

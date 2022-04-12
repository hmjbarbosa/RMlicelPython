import re
import numpy as np
from datetime import datetime


def profile_read(fname, dbin=0, dtime=0, ach=0, maxz=0):
    
    head = {}
    
    # Abrindo o arquivo
    
    with open(fname, 'r', encoding='utf8', errors='ignore') as fp:
        
        ## Linha 1
        regexp = re.compile('([\w]{9}.[\d]{3})')             # filename
        
        line = regexp.search(fp.readline())
        
        head['file'] = line.group(1)
        
        ## Linha 2
        regexp = re.compile(' ([\w_ ]*) '                    # site
                             '([\d]{2}/[\d]{2}/[\d]{4}) '    # datei
                             '([\d]{2}:[\d]{2}:[\d]{2}) '    # houri
                             '([\d]{2}/[\d]{2}/[\d]{4}) '    # datef
                             '([\d]{2}:[\d]{2}:[\d]{2}) '    # hourf
                             '([\d]{4}) '                    # alt
                             '(-?[\d]{3}\.\d) '              # lon
                             '(-?[\d]{3}\.\d) '              # lat
                             '(-?[\d]{2}) '                  # zen
                             '[\d]{2} '                      # ---- empty
                             '([\d]{2}\.\d) '                # T0
                             '([\d]{4}\.\d)')                # P0 
        
        line = regexp.search(fp.readline())
    
        head['site']  = line.group(1)
        head['datei'] = line.group(2)
        head['houri'] = line.group(3)
        head['datef'] = line.group(4)
        head['hourf'] = line.group(5)
        
        def datenum(d):    # Conversão da datenum do matlab pra python (Link [4])
            return 366 + d.toordinal() + (d - datetime.fromordinal(d.toordinal())).total_seconds()/(24*60*60)

        jdi = head['datei'] + ' ' + head['houri']
        jdi_strip = datetime.strptime(jdi, '%d/%m/%Y %H:%M:%S')
        
        jdf = head['datef'] + ' ' + head['hourf']
        jdf_strip = datetime.strptime(jdf, '%d/%m/%Y %H:%M:%S')
        
        head['jdi'] = datenum(jdi_strip)
        head['jdf'] = datenum(jdf_strip)
        
        head['alt'] = int(line.group(6))
        head['lon'] = float(line.group(7))
        head['lat'] = float(line.group(8))
        head['zen'] = float(line.group(9))
        head['T0']  = float(line.group(10)) 
        head['P0']  = float(line.group(11))
        
        ## Linha 3
        regexp = re.compile('([\d]{7}) '      # nshoots    
                            '([\d]{4}) '      # nhz
                            '([\d]{7}) '      # nshoots2
                            '([\d]{4}) '      # nhz2
                            '([\d]{2}) ')     # nch
        
        line = regexp.search(fp.readline())
        
        head['nshoots']  = int(line.group(1))
        head['nhz']      = int(line.group(2))
        head['nshoots2'] = int(line.group(3))
        head['nhz2']     = int(line.group(4))
        head['nch']      = int(line.group(5))
        
        ## Canais
        
        head['ch'] = {}
        nch = head['nch']            # Número de canais
        
        regexp = re.compile('(\d) '                # active
                            '(\d) '                # photons
                            '(\d) '                # elastic
                            '([\d]{5}) '           # ndata
                            '\d '                  # ----
                            '([\d]{4}) '           # pmtv
                            '(\d\.[\d]{2}) '       # binw
                            '([\d]{5})\.'          # wlen
                            '([osl]) '             # pol
                            '[0 ]{10} '            # ----
                            '([\d]{2}) '           # bits
                            '([\d]{6}) '           # nshoots
                            '(\d\.[\d]{3,4}) '     # discr
                            '([\w]{3})')           # tr

                            
        channels = ''.join([next(fp) for x in range(nch)])  # Aqui eu imprimo todos os canais
                                                            
        lines = np.array(regexp.findall(channels))
        
        head['ch']['active']  = lines[:, 0].astype(int)
        head['ch']['photons'] = lines[:, 1].astype(int)
        head['ch']['elastic'] = lines[:, 2].astype(int)
        head['ch']['ndata']   = lines[:, 3].astype(int)
        head['ch']['pmtv']    = lines[:, 4].astype(int)
        head['ch']['binw']    = lines[:, 5].astype(float)
        head['ch']['wlen']    = lines[:, 6].astype(int)
        head['ch']['pol']     = lines[:, 7]
        head['ch']['bits']    = lines[:, 8].astype(int)
        head['ch']['nshoots'] = lines[:, 9].astype(int)
        head['ch']['discr']   = lines[:, 10].astype(float)
        head['ch']['tr']      = lines[:, 11]
        
        # Criei os arrays phy e raw antes, pois no matlab elas são criadas enquanto declaradas
        
        max_linhas = max(head['ch']['ndata'])       # A solucao que encontrei aqui foi achar o max de
                                                    # linhas possivel que phy e raw podem ter para declarar antes        
        
        if ach == 0:
            phy = np.zeros((max_linhas, nch))
            raw = np.zeros((max_linhas, nch))
        else:
            phy = np.zeros((max_linhas, 1))
            raw = np.zeros((max_linhas, 1))            
 
        # conversion factor from raw to physical units
        for ch in range(nch):
            nz = head['ch']['ndata'][ch]
            trash=np.fromfile(fp, np.byte, 2)
            tmpraw = np.fromfile(fp, np.int32, nz)
            if ch == ach or ach == 0:
                if head['ch']['photons'][ch] == 0:
                    dScale = head['ch']['nshoots'][ch]*(2**head['ch']['bits'][ch])/(head['ch']['discr'][ch]*1e3)
                else:
                    dScale = head['ch']['nshoots'][ch]/20.
            
            tmpphy=tmpraw/dScale
            
            if ch == 1 or ch == 3:
                # displace by dbin's
                tmpphy[:nz-dbin] = tmpphy[dbin:nz]
                
                # repeat the last dbin values to keep size of vectors
                tmpphy[nz-dbin:nz] = tmpphy[nz-dbin:nz]
            else:
                # correct for dead-time
                tmpphy[:nz] = tmpphy[:nz]/(1-tmpphy[:nz]*dtime)
            
            # copy to final destination
            
            if maxz == 0:
                maxz = nz
            else:
                maxz = min(nz, maxz)
                head['ch']['ndata'][ch] = maxz
            
            if ach == 0:
                phy[:maxz, ch] = tmpphy[:maxz]
                raw[:maxz, ch] = tmpraw[:maxz]
            else:
                phy[:maxz] = tmpphy[:maxz]
                raw[:maxz] = tmpraw[:maxz]    
        
        
    return head, phy, raw
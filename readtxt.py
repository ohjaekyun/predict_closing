
with open('재무데이터.txt', 'r', encoding='euc-kr') as file:
    lines = file.readlines()

list_col = ['BIZ_NO', 'STT_YR_MO', 'M11000', 'M11130', 'M12000', 'M12300', 'M14900', 'M15000', 
            'M16000', 'M19900-M19800', 'M71000', 'M75000', 'M19800', 'M21000', 'M24000', 'M25000', 
            'M29000', 'M29500', 'M29900', 'MB1500', 'MB2700', 'MB2800', 'MB3100', 'MB3200', 'MB3400', 
            'MB3500', 'MB3510', 'MB3600', 'MB4200', 'MB5200', 'MB5300', 'MC1600', 'MC1700', 'MC1800', 
            'MC1900', 'MC2100', 'MC2200', 'MC3100', 'MD1100', 'MD1800', 'MD2300', 'MD2400', 'M11150', 
            'M22000', 'M12500', 'M11500']
with open('financial_data.csv', 'w', encoding='euc-kr') as file:
    for idx, line in enumerate(lines):
        if idx == 0:
            new_line = ','.join(list_col) + '\n'
        else:
            new_line = ','.join(line.split('\t'))
        file.write(new_line)

#data1['IS_LP'] = ((data1.CMP_SFIX_NM == '(자)').to_numpy() + (data1.CMP_PFIX_NM == '(자)').to_numpy())

#for ceo in ceonames:
#    if type(ceo) == str:
#        cn = ceo[-2]
#        try:
#            cns.append(int(cn) + 1)
#        except:
#            cns.append(1)
#    else:
#        cns.append(None)

#nhp=data1.HOMEPAGE_URL.to_numpy()
#for hp in nhp:
#    if type(hp) == str:
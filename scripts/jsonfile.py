import json

# Generate synthetic financial market Q&A data
data = [
    {"input": "O que é o índice Bovespa?", "output": "O índice Bovespa, ou Ibovespa, é o principal indicador de desempenho das ações negociadas na B3, a bolsa de valores do Brasil. Ele é composto pelas ações das empresas mais representativas do mercado brasileiro."},
    {"input": "Como funciona o mercado de ações?", "output": "O mercado de ações é um ambiente onde os investidores podem comprar e vender participações em empresas de capital aberto. As ações representam uma fração do capital social da empresa, e ao comprar uma ação, o investidor se torna um sócio da empresa."},
    {"input": "O que é uma corretora de valores?", "output": "Uma corretora de valores é uma instituição financeira que intermedia a compra e venda de ativos financeiros, como ações, títulos e fundos, no mercado financeiro."},
    {"input": "Qual a diferença entre renda fixa e renda variável?", "output": "Renda fixa refere-se a investimentos em que as condições de remuneração, como prazo e taxa de juros, são conhecidas no momento da aplicação, como CDBs e títulos públicos. Renda variável envolve investimentos onde a rentabilidade não é garantida, como ações e fundos imobiliários."},
    {"input": "O que é um fundo de investimento?", "output": "Um fundo de investimento é uma modalidade de aplicação financeira que reúne recursos de diversos investidores para serem aplicados em uma carteira diversificada de ativos, como ações, títulos e imóveis."},
    {"input": "Como funciona a tributação sobre investimentos em ações?", "output": "No Brasil, os lucros obtidos com a venda de ações são tributados pelo Imposto de Renda. O imposto é de 15% sobre os ganhos de capital e é responsabilidade do investidor calcular e pagar o imposto mensalmente através de DARF."},
    {"input": "O que é um IPO?", "output": "IPO, ou Initial Public Offering, é a oferta pública inicial de ações de uma empresa, ou seja, é a primeira vez que as ações da empresa são oferecidas ao público em uma bolsa de valores."},
    {"input": "O que é alavancagem no mercado financeiro?", "output": "Alavancagem é uma estratégia usada por investidores para aumentar seu potencial de retorno, utilizando recursos emprestados para amplificar os resultados de um investimento."},
    {"input": "O que significa diversificação de investimentos?", "output": "Diversificação de investimentos é uma estratégia que visa reduzir os riscos investindo em diferentes tipos de ativos e setores, ao invés de concentrar todo o capital em um único tipo de investimento."},
    {"input": "O que são derivativos?", "output": "Derivativos são instrumentos financeiros cujo valor deriva de outros ativos, como ações, índices ou commodities. Eles são usados para proteção contra oscilações de preço ou para especulação."},
    {"input": "Como funciona o mercado de câmbio?", "output": "O mercado de câmbio, ou Forex, é onde ocorrem as negociações de moedas estrangeiras. A taxa de câmbio é determinada pela oferta e demanda por cada moeda, e esse mercado é essencial para o comércio internacional."},
    {"input": "O que é um ETF?", "output": "ETF, ou Exchange Traded Fund, é um fundo de investimento negociado em bolsa, que busca replicar o desempenho de um índice de mercado, como o Ibovespa, por exemplo."},
    {"input": "O que é um CDB?", "output": "CDB, ou Certificado de Depósito Bancário, é um título de renda fixa emitido por bancos para captar recursos. O investidor empresta dinheiro ao banco e recebe em troca o valor investido acrescido de juros ao final do período de aplicação."},
    {"input": "O que é um título público?", "output": "Títulos públicos são instrumentos de dívida emitidos pelo governo para financiar suas atividades. No Brasil, são oferecidos através do Tesouro Direto e podem ter diferentes prazos e formas de remuneração."},
    {"input": "Qual o papel do Banco Central no mercado financeiro?", "output": "O Banco Central é a autoridade monetária responsável por garantir a estabilidade do sistema financeiro, controlar a inflação e regular a política monetária, além de supervisionar as instituições financeiras."},
    {"input": "O que é uma ação preferencial?", "output": "Ação preferencial é um tipo de ação que confere ao seu detentor preferência na distribuição de dividendos e, em alguns casos, no reembolso de capital, mas geralmente não dá direito a voto em assembleias."},
    {"input": "Como investir em ações no exterior?", "output": "Investir em ações no exterior pode ser feito através de corretoras que oferecem acesso a bolsas internacionais, ou por meio de BDRs, que são certificados representando ações de empresas estrangeiras negociados na B3."},
    {"input": "O que são small caps?", "output": "Small caps são ações de empresas com menor capitalização de mercado. Embora sejam consideradas mais arriscadas, essas ações podem oferecer um potencial de crescimento maior."},
    {"input": "O que é análise técnica?", "output": "Análise técnica é uma metodologia de estudo dos movimentos dos preços de ativos financeiros, utilizando gráficos e indicadores, para prever tendências futuras com base no histórico de negociações."},
    {"input": "O que é análise fundamentalista?", "output": "Análise fundamentalista é uma abordagem que avalia o valor intrínseco de uma empresa, considerando fatores econômicos, financeiros e de gestão, para determinar se uma ação está subvalorizada ou sobrevalorizada."}
]


output_file_no_unicode = 'financial_market_qa.jsonl'
with open(output_file_no_unicode, 'w', encoding='utf-8') as file:
    for entry in data:
        file.write(json.dumps(entry, ensure_ascii=False) + '\n')

output_file_no_unicode
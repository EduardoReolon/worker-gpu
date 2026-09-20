# O contrato do worker

Estes arquivos sao a **fonte da verdade** do que o worker devolve. Eles
existem porque o worker e os clientes moram em repositorios diferentes, e um
contrato que so vive na cabeca de quem escreveu diverge no primeiro mes.

    texto-resposta.json      POST /v1/chat/completions
    imagem-resposta.json     POST /v1/images/generations
    conversao-resposta.json  POST /parse/
    ocupada-resposta.json    qualquer rota, quando a GPU esta em uso
    saude-resposta.json      GET  /health/

Os codigos de `error.code` vivem no `ocupada-resposta.json`, com uma nota
por codigo dizendo se o cliente deve adiar ou desistir.

O `saude-resposta.json` chegou por ultimo, e a demora custou caro: o `/health/`
e o endpoint que TODO cliente consulta para diagnosticar, e era o unico sem
exemplo publicado. Quando as tres rotas viraram um servico so, o estado passou
a vir aninhado e `busy` virou `ocupada` na raiz — um cliente continuou lendo as
chaves antigas, `dict.get` devolveu `None`, e os avisos de diagnostico dele
sumiram sem erro nenhum, por semanas. Com o exemplo no lugar, uma mudanca de
forma quebra a suite DAQUI, e nao um cliente em producao.

## Como eles se mantem honestos

Dos dois lados, e por caminhos diferentes:

- **aqui**, `tests/test_contrato.py` gera uma resposta de verdade pelo HTTP e
  confere que a FORMA bate com o exemplo. Se alguem renomear um campo, o
  teste quebra neste repositorio;
- **no cliente**, o exemplo e copiado para os testes dele, e o adaptador e
  exercitado contra ele. Se o cliente passar a esperar outra coisa, o teste
  quebra la.

O que isso NAO pega e a copia do cliente ficar velha. Por isso todo exemplo
tem `"_contrato_versao"`: quando ela muda, quem integra precisa olhar.
Compare com:

    curl -s http://<worker>:8090/health/ | jq .contrato_versao

Uma mudanca que quebra compatibilidade sobe a versao MAIOR e vira uma secao
em `INTEGRACAO.md`. Acrescentar campo nao quebra ninguem e nao sobe nada:
todo cliente deve ignorar campo que nao conhece.

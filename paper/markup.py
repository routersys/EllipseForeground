def figure(identifier: str, source: str, description: str, caption: str, width: str = "100%") -> str:
    return (
        f'<figure id="{identifier}"><img src="figs/{source}" alt="{description}" style="width:{width}">'
        f'<figcaption><span class="no">図</span>{caption}</figcaption></figure>'
    )

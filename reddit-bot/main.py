"""Main entry point for the Reddit bot with CLI."""

import sys
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from loguru import logger

from src.bot import RedditBot

# Initialize CLI and console
app = typer.Typer(help="Reddit Automation Bot")
console = Console()

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("logs/bot.log", rotation="10 MB", retention="5 files", level="DEBUG")


@app.command()
def start(
    once: bool = typer.Option(False, "--once", help="Run once instead of continuously"),
    learn_style: bool = typer.Option(False, "--learn-style", help="Learn writing style first")
):
    """Start the Reddit bot."""
    try:
        console.print("[bold green]Starting Reddit Bot...[/bold green]")

        bot = RedditBot()

        # Learn writing style if requested
        if learn_style:
            console.print("[yellow]Learning your writing style from Reddit history...[/yellow]")
            bot.learn_writing_style()
            console.print("[green]✓ Writing style learned[/green]")

        # Run bot
        if once:
            console.print("[cyan]Running bot once...[/cyan]")
            bot.run_once()
            console.print("[green]✓ Bot run complete[/green]")
        else:
            console.print("[cyan]Starting continuous operation...[/cyan]")
            console.print("[yellow]Press Ctrl+C to stop[/yellow]\n")
            bot.run_continuous()

    except KeyboardInterrupt:
        console.print("\n[yellow]Bot stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Fatal error")
        sys.exit(1)


@app.command()
def review():
    """Review and approve pending replies."""
    try:
        bot = RedditBot()
        pending = bot.get_pending_replies()

        if not pending:
            console.print("[green]No pending replies to review[/green]")
            return

        console.print(f"[cyan]Found {len(pending)} pending replies[/cyan]\n")

        for i, reply in enumerate(pending):
            # Display reply info
            panel_content = f"""
[bold]Subreddit:[/bold] r/{reply['subreddit']}
[bold]Title:[/bold] {reply['title'][:80]}
[bold]URL:[/bold] {reply['url']}
[bold]Confidence:[/bold] {reply['confidence']:.2f}

[bold]Proposed Reply:[/bold]
{reply['response']}
            """

            console.print(Panel(panel_content, title=f"Reply #{i+1}", border_style="cyan"))

            # Ask for action
            action = Prompt.ask(
                "\nAction",
                choices=["approve", "edit", "reject", "skip", "quit"],
                default="skip"
            )

            if action == "approve":
                result = bot.approve_reply(i)
                if result.get('status') == 'posted':
                    console.print("[green]✓ Reply posted successfully![/green]\n")
                else:
                    console.print(f"[red]✗ Failed: {result.get('error')}[/red]\n")

            elif action == "edit":
                console.print("\n[yellow]Enter edited reply (Ctrl+D when done):[/yellow]")
                edited_lines = []
                try:
                    while True:
                        line = input()
                        edited_lines.append(line)
                except EOFError:
                    pass

                edited_text = "\n".join(edited_lines)
                result = bot.edit_and_approve_reply(i, edited_text)

                if result.get('status') == 'posted':
                    console.print("[green]✓ Edited reply posted successfully![/green]\n")
                else:
                    console.print(f"[red]✗ Failed: {result.get('error')}[/red]\n")

            elif action == "reject":
                bot.reject_reply(i)
                console.print("[yellow]Reply rejected[/yellow]\n")

            elif action == "quit":
                break

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Error in review")


@app.command()
def stats():
    """Show bot statistics."""
    try:
        bot = RedditBot()
        stats = bot.get_stats()

        table = Table(title="Bot Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Posts Seen", str(stats['posts_seen']))
        table.add_row("Comments Seen", str(stats['comments_seen']))
        table.add_row("Total Replies", str(stats['total_replies']))
        table.add_row("Replies Today", str(stats['replies_today']))
        table.add_row("User Activities Tracked", str(stats['user_activities_tracked']))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def post(
    subreddit: str = typer.Argument(..., help="Subreddit to post in"),
    title: str = typer.Argument(..., help="Post title"),
    text: str = typer.Option(None, "--text", "-t", help="Post text content"),
    url: str = typer.Option(None, "--url", "-u", help="Post URL (for link posts)")
):
    """Create a Reddit post."""
    try:
        if not text and not url:
            console.print("[red]Error: Must provide either --text or --url[/red]")
            return

        bot = RedditBot()

        console.print(f"[cyan]Creating post in r/{subreddit}...[/cyan]")

        post_id = bot.create_post(subreddit, title, text=text, url=url)

        if post_id:
            console.print(f"[green]✓ Post created successfully![/green]")
            console.print(f"[cyan]Post ID: {post_id}[/cyan]")
        else:
            console.print("[red]✗ Failed to create post[/red]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Error creating post")


@app.command()
def learn_style():
    """Learn your writing style from Reddit comment history."""
    try:
        console.print("[cyan]Analyzing your Reddit comment history...[/cyan]")

        bot = RedditBot()
        bot.learn_writing_style()

        console.print("[green]✓ Writing style learned successfully![/green]")
        console.print("[yellow]The bot will now mimic your writing style in replies[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Error learning style")


@app.command()
def test_auth():
    """Test Reddit API authentication."""
    try:
        from src.reddit_client import RedditClient

        console.print("[cyan]Testing Reddit authentication...[/cyan]")

        client = RedditClient()

        if client.is_authenticated():
            console.print(f"[green]✓ Successfully authenticated as u/{client.username}[/green]")
        else:
            console.print("[red]✗ Authentication failed[/red]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Authentication error")


if __name__ == "__main__":
    app()

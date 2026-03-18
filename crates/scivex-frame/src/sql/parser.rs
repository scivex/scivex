//! Recursive-descent SQL parser.

use crate::error::{FrameError, Result};

use super::ast::{
    BinaryOp, JoinClause, JoinKind, OrderItem, SelectItem, SelectStatement, SqlExpr, SqlLiteral,
    TableRef,
};
use super::tokenizer::Token;

/// Parser state: a cursor over a token slice.
struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> &Token {
        let tok = self.tokens.get(self.pos).unwrap_or(&Token::Eof);
        self.pos += 1;
        tok
    }

    fn expect(&mut self, expected: &Token) -> Result<()> {
        let tok = self.advance().clone();
        if std::mem::discriminant(&tok) == std::mem::discriminant(expected) {
            Ok(())
        } else {
            Err(FrameError::InvalidValue {
                reason: format!("expected {expected:?}, got {tok:?}"),
            })
        }
    }

    fn expect_ident(&mut self) -> Result<String> {
        match self.advance().clone() {
            Token::Ident(s) => Ok(s),
            other => Err(FrameError::InvalidValue {
                reason: format!("expected identifier, got {other:?}"),
            }),
        }
    }

    /// Check if the current token matches without consuming it.
    fn check(&self, token: &Token) -> bool {
        std::mem::discriminant(self.peek()) == std::mem::discriminant(token)
    }

    /// Consume the current token if it matches, returning true.
    fn eat(&mut self, token: &Token) -> bool {
        if self.check(token) {
            self.advance();
            true
        } else {
            false
        }
    }
}

/// Parse a token stream into a `SelectStatement`.
pub fn parse(tokens: &[Token]) -> Result<SelectStatement> {
    let mut parser = Parser::new(tokens);
    let stmt = parse_select(&mut parser)?;
    // Allow trailing EOF
    if !parser.check(&Token::Eof) {
        return Err(FrameError::InvalidValue {
            reason: format!("unexpected token after statement: {:?}", parser.peek()),
        });
    }
    Ok(stmt)
}

fn parse_select(p: &mut Parser<'_>) -> Result<SelectStatement> {
    p.expect(&Token::Select)?;

    // Parse projections
    let projections = parse_select_list(p)?;

    // FROM
    p.expect(&Token::From)?;
    let from = parse_from_list(p)?;

    // JOINs
    let joins = parse_joins(p)?;

    // WHERE
    let where_clause = if p.eat(&Token::Where) {
        Some(parse_expr(p)?)
    } else {
        None
    };

    // GROUP BY
    let group_by = if p.eat(&Token::Group) {
        p.expect(&Token::By)?;
        parse_expr_list(p)?
    } else {
        Vec::new()
    };

    // HAVING
    let having = if p.eat(&Token::Having) {
        Some(parse_expr(p)?)
    } else {
        None
    };

    // ORDER BY
    let order_by = if p.eat(&Token::Order) {
        p.expect(&Token::By)?;
        parse_order_list(p)?
    } else {
        Vec::new()
    };

    // LIMIT
    let limit = if p.eat(&Token::Limit) {
        match p.advance().clone() {
            Token::Integer(n) => {
                if n < 0 {
                    return Err(FrameError::InvalidValue {
                        reason: "LIMIT must be non-negative".to_string(),
                    });
                }
                Some(n as usize)
            }
            other => {
                return Err(FrameError::InvalidValue {
                    reason: format!("expected integer after LIMIT, got {other:?}"),
                });
            }
        }
    } else {
        None
    };

    Ok(SelectStatement {
        projections,
        from,
        joins,
        where_clause,
        group_by,
        having,
        order_by,
        limit,
    })
}

fn parse_select_list(p: &mut Parser<'_>) -> Result<Vec<SelectItem>> {
    let mut items = Vec::new();

    // Check for SELECT *
    if p.check(&Token::Star) {
        p.advance();
        items.push(SelectItem::Wildcard);
        // Could be `SELECT *, expr` but we don't support that — just return
        return Ok(items);
    }

    loop {
        let expr = parse_expr(p)?;
        let alias = if p.eat(&Token::As) {
            Some(parse_ident_or_keyword(p)?)
        } else {
            None
        };
        items.push(SelectItem::Expr { expr, alias });
        if !p.eat(&Token::Comma) {
            break;
        }
    }
    Ok(items)
}

/// Parse an identifier, also accepting aggregate keywords used as aliases.
fn parse_ident_or_keyword(p: &mut Parser<'_>) -> Result<String> {
    match p.advance().clone() {
        Token::Ident(s) => Ok(s),
        Token::Sum => Ok("SUM".to_string()),
        Token::Avg => Ok("AVG".to_string()),
        Token::Min => Ok("MIN".to_string()),
        Token::Max => Ok("MAX".to_string()),
        Token::Count => Ok("COUNT".to_string()),
        other => Err(FrameError::InvalidValue {
            reason: format!("expected identifier, got {other:?}"),
        }),
    }
}

fn parse_from_list(p: &mut Parser<'_>) -> Result<Vec<TableRef>> {
    let mut tables = Vec::new();
    loop {
        let name = parse_ident_or_keyword(p)?;
        let alias = if p.check(&Token::Ident(String::new())) {
            Some(p.expect_ident()?)
        } else if p.eat(&Token::As) {
            Some(parse_ident_or_keyword(p)?)
        } else {
            None
        };
        tables.push(TableRef { name, alias });
        if !p.eat(&Token::Comma) {
            break;
        }
    }
    Ok(tables)
}

fn parse_joins(p: &mut Parser<'_>) -> Result<Vec<JoinClause>> {
    let mut joins = Vec::new();
    loop {
        let join_type = match p.peek() {
            Token::Inner => {
                p.advance();
                p.expect(&Token::Join)?;
                JoinKind::Inner
            }
            Token::Left => {
                p.advance();
                // optional OUTER
                p.eat(&Token::Outer);
                p.expect(&Token::Join)?;
                JoinKind::Left
            }
            Token::Right => {
                p.advance();
                // optional OUTER
                p.eat(&Token::Outer);
                p.expect(&Token::Join)?;
                JoinKind::Right
            }
            Token::Join => {
                p.advance();
                JoinKind::Inner // bare JOIN = INNER JOIN
            }
            _ => break,
        };

        let tname = parse_ident_or_keyword(p)?;
        let talias = if p.check(&Token::Ident(String::new())) && !matches!(p.peek(), Token::On) {
            // Peek ahead: this is an alias if next token is not ON.
            // But we cannot easily peek ahead two tokens, so use the ON check.
            Some(p.expect_ident()?)
        } else if p.eat(&Token::As) {
            Some(parse_ident_or_keyword(p)?)
        } else {
            None
        };

        p.expect(&Token::On)?;
        let on = parse_expr(p)?;

        joins.push(JoinClause {
            join_type,
            table: TableRef {
                name: tname,
                alias: talias,
            },
            on,
        });
    }
    Ok(joins)
}

fn parse_expr_list(p: &mut Parser<'_>) -> Result<Vec<SqlExpr>> {
    let mut exprs = Vec::new();
    loop {
        exprs.push(parse_expr(p)?);
        if !p.eat(&Token::Comma) {
            break;
        }
    }
    Ok(exprs)
}

fn parse_order_list(p: &mut Parser<'_>) -> Result<Vec<OrderItem>> {
    let mut items = Vec::new();
    loop {
        let expr = parse_expr(p)?;
        let ascending = if p.eat(&Token::Desc) {
            false
        } else {
            p.eat(&Token::Asc);
            true
        };
        items.push(OrderItem { expr, ascending });
        if !p.eat(&Token::Comma) {
            break;
        }
    }
    Ok(items)
}

// ---------------------------------------------------------------------------
// Expression parsing with precedence
// ---------------------------------------------------------------------------
// Precedence (low to high): OR, AND, NOT, comparison, add/sub, mul/div, unary, primary

fn parse_expr(p: &mut Parser<'_>) -> Result<SqlExpr> {
    parse_or(p)
}

fn parse_or(p: &mut Parser<'_>) -> Result<SqlExpr> {
    let mut left = parse_and(p)?;
    while p.eat(&Token::Or) {
        let right = parse_and(p)?;
        left = SqlExpr::BinaryOp {
            left: Box::new(left),
            op: BinaryOp::Or,
            right: Box::new(right),
        };
    }
    Ok(left)
}

fn parse_and(p: &mut Parser<'_>) -> Result<SqlExpr> {
    let mut left = parse_not(p)?;
    while p.eat(&Token::And) {
        let right = parse_not(p)?;
        left = SqlExpr::BinaryOp {
            left: Box::new(left),
            op: BinaryOp::And,
            right: Box::new(right),
        };
    }
    Ok(left)
}

fn parse_not(p: &mut Parser<'_>) -> Result<SqlExpr> {
    if p.eat(&Token::Not) {
        let expr = parse_not(p)?;
        Ok(SqlExpr::UnaryNot(Box::new(expr)))
    } else {
        parse_comparison(p)
    }
}

fn parse_comparison(p: &mut Parser<'_>) -> Result<SqlExpr> {
    let mut left = parse_addition(p)?;

    loop {
        let op = match p.peek() {
            Token::Eq => BinaryOp::Eq,
            Token::NotEq => BinaryOp::NotEq,
            Token::Lt => BinaryOp::Lt,
            Token::LtEq => BinaryOp::LtEq,
            Token::Gt => BinaryOp::Gt,
            Token::GtEq => BinaryOp::GtEq,
            Token::Is => {
                p.advance();
                p.expect(&Token::Null)?;
                left = SqlExpr::IsNull(Box::new(left));
                continue;
            }
            _ => break,
        };
        p.advance();
        let right = parse_addition(p)?;
        left = SqlExpr::BinaryOp {
            left: Box::new(left),
            op,
            right: Box::new(right),
        };
    }
    Ok(left)
}

fn parse_addition(p: &mut Parser<'_>) -> Result<SqlExpr> {
    let mut left = parse_multiplication(p)?;
    loop {
        let op = match p.peek() {
            Token::Plus => BinaryOp::Plus,
            Token::Minus => BinaryOp::Minus,
            _ => break,
        };
        p.advance();
        let right = parse_multiplication(p)?;
        left = SqlExpr::BinaryOp {
            left: Box::new(left),
            op,
            right: Box::new(right),
        };
    }
    Ok(left)
}

fn parse_multiplication(p: &mut Parser<'_>) -> Result<SqlExpr> {
    let mut left = parse_primary(p)?;
    loop {
        let op = match p.peek() {
            Token::Star => BinaryOp::Mul,
            Token::Slash => BinaryOp::Div,
            _ => break,
        };
        p.advance();
        let right = parse_primary(p)?;
        left = SqlExpr::BinaryOp {
            left: Box::new(left),
            op,
            right: Box::new(right),
        };
    }
    Ok(left)
}

fn parse_primary(p: &mut Parser<'_>) -> Result<SqlExpr> {
    match p.peek().clone() {
        Token::Integer(n) => {
            p.advance();
            Ok(SqlExpr::Literal(SqlLiteral::Integer(n)))
        }
        Token::Float(f) => {
            p.advance();
            Ok(SqlExpr::Literal(SqlLiteral::Float(f)))
        }
        Token::StringLit(s) => {
            p.advance();
            Ok(SqlExpr::Literal(SqlLiteral::String(s)))
        }
        Token::Null => {
            p.advance();
            Ok(SqlExpr::Literal(SqlLiteral::Null))
        }
        Token::Star => {
            p.advance();
            Ok(SqlExpr::Wildcard)
        }
        Token::LParen => {
            p.advance();
            let expr = parse_expr(p)?;
            p.expect(&Token::RParen)?;
            Ok(expr)
        }
        Token::Sum | Token::Avg | Token::Min | Token::Max | Token::Count => {
            let name = match p.advance().clone() {
                Token::Sum => "SUM",
                Token::Avg => "AVG",
                Token::Min => "MIN",
                Token::Max => "MAX",
                Token::Count => "COUNT",
                _ => unreachable!(),
            }
            .to_string();
            p.expect(&Token::LParen)?;
            let mut args = Vec::new();
            if !p.check(&Token::RParen) {
                loop {
                    args.push(parse_expr(p)?);
                    if !p.eat(&Token::Comma) {
                        break;
                    }
                }
            }
            p.expect(&Token::RParen)?;
            Ok(SqlExpr::Function { name, args })
        }
        Token::Ident(name) => {
            p.advance();
            // Check for qualified column: ident.ident
            if p.eat(&Token::Dot) {
                // Could be ident.* or ident.column
                if p.check(&Token::Star) {
                    p.advance();
                    // table.* — treat as qualified wildcard; we'll handle in executor
                    Ok(SqlExpr::QualifiedColumn {
                        table: name,
                        column: "*".to_string(),
                    })
                } else {
                    let col = parse_ident_or_keyword(p)?;
                    Ok(SqlExpr::QualifiedColumn {
                        table: name,
                        column: col,
                    })
                }
            } else if p.check(&Token::LParen) {
                // User-named function (though we don't support arbitrary UDFs,
                // treat it as a function call).
                p.advance(); // consume LParen
                let mut args = Vec::new();
                if !p.check(&Token::RParen) {
                    loop {
                        args.push(parse_expr(p)?);
                        if !p.eat(&Token::Comma) {
                            break;
                        }
                    }
                }
                p.expect(&Token::RParen)?;
                Ok(SqlExpr::Function {
                    name: name.to_uppercase(),
                    args,
                })
            } else {
                Ok(SqlExpr::Column(name))
            }
        }
        other => Err(FrameError::InvalidValue {
            reason: format!("unexpected token in expression: {other:?}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::tokenizer::tokenize;

    #[test]
    fn test_parse_select_star() {
        let tokens = tokenize("SELECT * FROM t").unwrap();
        let stmt = parse(&tokens).unwrap();
        assert_eq!(stmt.projections.len(), 1);
        assert!(matches!(stmt.projections[0], SelectItem::Wildcard));
        assert_eq!(stmt.from.len(), 1);
        assert_eq!(stmt.from[0].name, "t");
    }

    #[test]
    fn test_parse_select_columns() {
        let tokens = tokenize("SELECT a, b FROM t").unwrap();
        let stmt = parse(&tokens).unwrap();
        assert_eq!(stmt.projections.len(), 2);
    }

    #[test]
    fn test_parse_where() {
        let tokens = tokenize("SELECT * FROM t WHERE a > 5").unwrap();
        let stmt = parse(&tokens).unwrap();
        assert!(stmt.where_clause.is_some());
    }

    #[test]
    fn test_parse_group_by() {
        let tokens = tokenize("SELECT g, SUM(v) FROM t GROUP BY g").unwrap();
        let stmt = parse(&tokens).unwrap();
        assert_eq!(stmt.group_by.len(), 1);
    }

    #[test]
    fn test_parse_order_limit() {
        let tokens = tokenize("SELECT * FROM t ORDER BY a DESC LIMIT 10").unwrap();
        let stmt = parse(&tokens).unwrap();
        assert_eq!(stmt.order_by.len(), 1);
        assert!(!stmt.order_by[0].ascending);
        assert_eq!(stmt.limit, Some(10));
    }

    #[test]
    fn test_parse_join() {
        let tokens = tokenize("SELECT * FROM t1 INNER JOIN t2 ON t1.id = t2.id").unwrap();
        let stmt = parse(&tokens).unwrap();
        assert_eq!(stmt.joins.len(), 1);
        assert_eq!(stmt.joins[0].join_type, JoinKind::Inner);
    }

    #[test]
    fn test_parse_error_no_from() {
        let tokens = tokenize("SELECT *").unwrap();
        assert!(parse(&tokens).is_err());
    }
}

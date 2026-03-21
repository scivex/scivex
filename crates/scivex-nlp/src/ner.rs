use std::collections::{HashMap, HashSet};
use std::fmt;

/// Named entity type variants.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::EntityType;
/// let t = EntityType::Person;
/// assert_eq!(format!("{t}"), "Person");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Date,
    Number,
    Other,
}

impl fmt::Display for EntityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            EntityType::Person => "Person",
            EntityType::Organization => "Organization",
            EntityType::Location => "Location",
            EntityType::Date => "Date",
            EntityType::Number => "Number",
            EntityType::Other => "Other",
        };
        write!(f, "{s}")
    }
}

/// A recognized named entity with its type and position in the token sequence.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::ner::{Entity, EntityType};
/// let e = Entity { text: "London".to_string(), entity_type: EntityType::Location, start: 0, end: 1 };
/// assert_eq!(e.entity_type, EntityType::Location);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entity {
    /// The entity text.
    pub text: String,
    /// The entity type.
    pub entity_type: EntityType,
    /// Token start index (inclusive).
    pub start: usize,
    /// Token end index (exclusive).
    pub end: usize,
}

/// Rule-based named entity recognizer using gazetteers and pattern rules.
///
/// # Examples
///
/// ```
/// # use scivex_nlp::ner::{RuleBasedNer, EntityType};
/// let mut ner = RuleBasedNer::new();
/// ner.add_entity(EntityType::Person, "Alice");
/// let entities = ner.recognize(&["Alice", "went", "home"]);
/// assert_eq!(entities.len(), 1);
/// ```
pub struct RuleBasedNer {
    /// Known entity lists keyed by entity type.
    gazetteers: HashMap<EntityType, HashSet<String>>,
}

/// Month names used for date detection.
const MONTH_NAMES: [&str; 12] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

impl Default for RuleBasedNer {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleBasedNer {
    /// Create a new rule-based NER with built-in gazetteers for persons,
    /// organizations, and locations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::ner::RuleBasedNer;
    /// let ner = RuleBasedNer::new();
    /// let entities = ner.recognize(&["John", "works", "at", "Google"]);
    /// assert!(entities.iter().any(|e| e.text == "John"));
    /// assert!(entities.iter().any(|e| e.text == "Google"));
    /// ```
    pub fn new() -> Self {
        let mut gazetteers: HashMap<EntityType, HashSet<String>> = HashMap::new();

        let persons: HashSet<String> = [
            "John", "Mary", "James", "Robert", "Michael", "William", "David", "Richard", "Joseph",
            "Thomas",
        ]
        .iter()
        .map(std::string::ToString::to_string)
        .collect();

        let organizations: HashSet<String> = [
            "Google",
            "Apple",
            "Microsoft",
            "Amazon",
            "Facebook",
            "NASA",
            "FBI",
            "CIA",
            "UN",
            "NATO",
        ]
        .iter()
        .map(std::string::ToString::to_string)
        .collect();

        let locations: HashSet<String> = [
            "London",
            "Paris",
            "Tokyo",
            "Berlin",
            "Rome",
            "Moscow",
            "Beijing",
            "Washington",
            "York",
            "California",
        ]
        .iter()
        .map(std::string::ToString::to_string)
        .collect();

        gazetteers.insert(EntityType::Person, persons);
        gazetteers.insert(EntityType::Organization, organizations);
        gazetteers.insert(EntityType::Location, locations);

        RuleBasedNer { gazetteers }
    }

    /// Add an entity to a gazetteer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::ner::{RuleBasedNer, EntityType};
    /// let mut ner = RuleBasedNer::new();
    /// ner.add_entity(EntityType::Organization, "Scivex");
    /// let entities = ner.recognize(&["use", "Scivex"]);
    /// assert!(entities.iter().any(|e| e.text == "Scivex"));
    /// ```
    pub fn add_entity(&mut self, entity_type: EntityType, name: &str) {
        self.gazetteers
            .entry(entity_type)
            .or_default()
            .insert(name.to_string());
    }

    /// Recognize named entities in a token sequence.
    ///
    /// Applies gazetteer lookup, capitalization heuristics, month name detection,
    /// and number pattern matching.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_nlp::ner::{RuleBasedNer, EntityType};
    /// let ner = RuleBasedNer::new();
    /// let entities = ner.recognize(&["born", "in", "January", "42"]);
    /// assert!(entities.iter().any(|e| e.entity_type == EntityType::Date));
    /// assert!(entities.iter().any(|e| e.entity_type == EntityType::Number));
    /// ```
    pub fn recognize(&self, tokens: &[&str]) -> Vec<Entity> {
        let mut entities = Vec::new();

        for (i, &token) in tokens.iter().enumerate() {
            // Strip trailing punctuation for matching
            let clean = token.trim_end_matches(|c: char| c.is_ascii_punctuation());
            if clean.is_empty() {
                continue;
            }

            // Check gazetteers first (exact match)
            let mut found = false;
            // Check in a deterministic order: Person, Organization, Location
            for entity_type in &[
                EntityType::Person,
                EntityType::Organization,
                EntityType::Location,
            ] {
                let in_gazetteer = self
                    .gazetteers
                    .get(entity_type)
                    .is_some_and(|set| set.contains(clean));
                if in_gazetteer {
                    entities.push(Entity {
                        text: clean.to_string(),
                        entity_type: entity_type.clone(),
                        start: i,
                        end: i + 1,
                    });
                    found = true;
                    break;
                }
            }
            if found {
                continue;
            }

            // Check for month names (Date entity)
            if MONTH_NAMES.contains(&clean.to_lowercase().as_str()) {
                entities.push(Entity {
                    text: clean.to_string(),
                    entity_type: EntityType::Date,
                    start: i,
                    end: i + 1,
                });
                continue;
            }

            // Check for number patterns
            let is_number = clean.parse::<f64>().is_ok();
            if is_number {
                entities.push(Entity {
                    text: clean.to_string(),
                    entity_type: EntityType::Number,
                    start: i,
                    end: i + 1,
                });
                continue;
            }

            // Capitalization heuristic: capitalized word not at sentence start
            // could be a Person or Organization
            let is_capitalized =
                i > 0 && clean.len() > 1 && clean.chars().next().is_some_and(char::is_uppercase);
            if is_capitalized {
                // Check if previous token ended a sentence
                let prev = tokens[i - 1];
                let prev_ends_sentence =
                    prev.ends_with('.') || prev.ends_with('!') || prev.ends_with('?');
                if !prev_ends_sentence {
                    entities.push(Entity {
                        text: clean.to_string(),
                        entity_type: EntityType::Other,
                        start: i,
                        end: i + 1,
                    });
                }
            }
        }

        entities
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_entities() {
        let ner = RuleBasedNer::new();
        let tokens = vec!["John", "works", "at", "Google", "in", "London"];
        let entities = ner.recognize(&tokens);

        let john = entities.iter().find(|e| e.text == "John").unwrap();
        assert_eq!(john.entity_type, EntityType::Person);
        assert_eq!(john.start, 0);
        assert_eq!(john.end, 1);

        let google = entities.iter().find(|e| e.text == "Google").unwrap();
        assert_eq!(google.entity_type, EntityType::Organization);

        let london = entities.iter().find(|e| e.text == "London").unwrap();
        assert_eq!(london.entity_type, EntityType::Location);
    }

    #[test]
    fn test_date_month_detection() {
        let ner = RuleBasedNer::new();
        let tokens = vec!["born", "in", "January"];
        let entities = ner.recognize(&tokens);

        let month = entities.iter().find(|e| e.text == "January").unwrap();
        assert_eq!(month.entity_type, EntityType::Date);
    }

    #[test]
    fn test_number_detection() {
        let ner = RuleBasedNer::new();
        let tokens = vec!["there", "are", "42", "items"];
        let entities = ner.recognize(&tokens);

        let num = entities.iter().find(|e| e.text == "42").unwrap();
        assert_eq!(num.entity_type, EntityType::Number);
        assert_eq!(num.start, 2);
        assert_eq!(num.end, 3);
    }

    #[test]
    fn test_custom_entity() {
        let mut ner = RuleBasedNer::new();
        ner.add_entity(EntityType::Organization, "Scivex");
        let tokens = vec!["use", "Scivex", "today"];
        let entities = ner.recognize(&tokens);

        let scivex = entities.iter().find(|e| e.text == "Scivex").unwrap();
        assert_eq!(scivex.entity_type, EntityType::Organization);
    }
}

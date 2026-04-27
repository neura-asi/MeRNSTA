CREATE TABLE enhanced_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL DEFAULT 0.8,
                contradiction BOOLEAN DEFAULT FALSE,
                volatile BOOLEAN DEFAULT FALSE,
                volatility_score REAL DEFAULT 0.0,
                user_profile_id TEXT,
                session_id TEXT,
                source_message_id INTEGER,
                hedge_detected BOOLEAN DEFAULT FALSE,
                intensifier_detected BOOLEAN DEFAULT FALSE,
                negation BOOLEAN DEFAULT FALSE,
                embedding TEXT,
                change_history TEXT,
                active BOOLEAN DEFAULT TRUE,
                UNIQUE(subject, predicate, object, user_profile_id, session_id)
            );
CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE contradiction_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_a_id INTEGER,
                fact_b_id INTEGER,
                timestamp TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_notes TEXT,
                confidence REAL DEFAULT 1.0,
                FOREIGN KEY (fact_a_id) REFERENCES enhanced_facts(id),
                FOREIGN KEY (fact_b_id) REFERENCES enhanced_facts(id)
            );
CREATE INDEX idx_facts_user ON enhanced_facts(user_profile_id);
CREATE INDEX idx_facts_session ON enhanced_facts(session_id);
CREATE INDEX idx_facts_subject_predicate ON enhanced_facts(subject, predicate);
CREATE INDEX idx_facts_timestamp ON enhanced_facts(timestamp);

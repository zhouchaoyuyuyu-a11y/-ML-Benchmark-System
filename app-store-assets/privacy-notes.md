# Privacy questionnaires — App Store & Google Play

Grounded in the actual data model (`db/schema.sql`): the platform stores
accounts, self-expression profiles, relationship profiles, conversations,
drafts, archives, orders, and AI usage logs. There is no third-party ad SDK,
no cross-app tracking, and no sale of data. Payment card data never touches
the platform — gateways (Stripe / WeChat Pay / Alipay / PayPal) hold it, and
when no gateway is configured, orders route to human concierge confirmation
with no payment collected in-app.

The authoritative user-facing texts live at `/legal/privacy` (Privacy
Policy v2.3) and `/legal/app` (App Privacy Notice v1.0).

---

## App Store — privacy nutrition labels

Declare **Data Linked to You** for the following (everything is account-bound
once signed in). Nothing is declared under "Data Used to Track You" — the app
does not track across companies, so App Tracking Transparency is not
required.

| Apple category | Data | Source table(s) | Purpose(s) |
| --- | --- | --- | --- |
| Contact Info → Email Address | Account email | `users.email` | App Functionality |
| Contact Info → Phone Number | Optional phone | `users.phone` | App Functionality |
| Contact Info → Name | Nickname / lead contact name | `users.nickname`, `concierge_leads.name` | App Functionality |
| Identifiers → User ID | Account id, visitor id for guest quota | `users.id`, `conversations.visitor_id` | App Functionality |
| Purchases → Purchase History | Orders (membership, casting deposits, …) | `orders` | App Functionality |
| User Content → Other User Content | Concierge messages, drafts, profiles (incl. optional MBTI/zodiac style tags), relationship notes, co-creation concepts, archive records | `messages`, `object_drafts`, `user_profiles`, `relationship_profiles`, `co_creation_projects`, `reserve_records` | App Functionality, Product Personalization |
| Usage Data → Product Interaction | AI quota/usage accounting | `ai_usage_logs` | App Functionality, Analytics |

Declared as **not collected**: precise location, contacts, photos/videos,
health & fitness, financial info (card numbers), browsing history, search
history (outside the app), sensitive info in Apple's sense (the optional
MBTI/zodiac/blood-type fields are self-expression style tags, declared under
User Content; they are optional, editable, deletable, and private by
default).

Third parties: the configured AI provider (Anthropic or OpenAI) processes
conversation text to generate proposals — declare under "Data collected by
third-party partners" with purpose App Functionality. With no provider
configured, generation runs locally on the platform (Atelier engine) and no
text leaves it.

## Google Play — Data safety form

| Play data type | Collected | Shared | Optional | Purpose |
| --- | --- | --- | --- | --- |
| Personal info → Name | Yes | No | Yes | App functionality |
| Personal info → Email address | Yes | No | No (needed for account) | App functionality, Account management |
| Personal info → Phone number | Yes | No | Yes | App functionality |
| Financial info → Purchase history | Yes | No | Yes | App functionality |
| Messages → Other in-app messages | Yes | No* | Yes | App functionality, Personalization |
| App activity → App interactions | Yes | No | No | Analytics, App functionality |
| App activity → Other user-generated content | Yes | No | Yes | App functionality, Personalization |
| Device or other IDs | Yes (visitor id for guest quota) | No | No | App functionality |

\* Concierge message text is processed by the configured AI provider as a
service provider (processor), which Play's form treats as collection, not
"sharing", when bound by contract — keep the provider under your data
processing agreement and answer "shared: No".

Security answers: data encrypted in transit (TLS everywhere); users can
request deletion (in-app + web, see below); the app targets an 18+ audience
(no Families policy obligations); independent security review: not enrolled.

## Alcohol content disclosure

- Both questionnaires: declare alcohol **references** (see
  `app-name.md` for the age-rating answers: Apple 17+, IARC per region).
- The app does not sell alcohol in-app. AI results are creative proposals;
  physical production and delivery require human concierge confirmation with
  age verification and regional compliance checks (Alcohol Compliance Notice
  v1.5 at `/legal/alcohol`, Minor Protection Notice v1.1 at `/legal/minors`).
- An 18+ age gate covers alcohol-related screens (`age_gate_enabled`
  platform setting). Screenshots contain no consumption imagery.

## Account deletion

- **In-app path:** Profile → account section (`/profile`) — satisfies
  Apple's account-deletion requirement (the deletion entry is reachable
  inside the app, not only on the web).
- **Web URL for the Play Console "data deletion" field:**
  `https://<NEXT_PUBLIC_SITE_URL>/profile` (e.g.
  `https://zotaix-web.vercel.app/profile`).
- Deletion removes the account and personal profiles; public co-creation
  projects a user founded are detached and handled per the Co-Creation Pool
  Rules (`/legal/co-create`); Reserve records owned by the account are
  deleted with it unless the user exports them first.
- Data retention: quota/usage logs are kept in aggregate for billing
  integrity; personal identifiers are removed on deletion.

## Contact points for both consoles

- Privacy policy URL: `https://<NEXT_PUBLIC_SITE_URL>/legal/privacy`
- App privacy notice: `https://<NEXT_PUBLIC_SITE_URL>/legal/app`
- Support/contact: `https://<NEXT_PUBLIC_SITE_URL>/legal/contact`, or the
  operations mailbox configured in `ADMIN_EMAIL`.

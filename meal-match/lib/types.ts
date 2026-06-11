// Domain model for Meal Match — a home-cook marketplace where diners are
// matched both with a cook and with "meal buddies" to share the table.

export type Cuisine =
  | "italian"
  | "indian"
  | "japanese"
  | "mexican"
  | "lebanese"
  | "thai"
  | "ethiopian"
  | "french"
  | "korean"
  | "vegan-soul";

export type Dietary =
  | "vegetarian"
  | "vegan"
  | "halal"
  | "gluten-free"
  | "nut-free"
  | "dairy-free";

/** Coarse meal sittings used for time-window matching. */
export type Sitting = "lunch" | "early-dinner" | "late-dinner";

export interface Cook {
  id: string;
  name: string;
  avatar: string; // emoji stand-in for a profile photo
  neighborhood: string;
  cuisines: Cuisine[];
  rating: number; // 0..5
  reviews: number;
  bio: string;
  signatureDish: string;
}

export interface Meal {
  id: string;
  cookId: string;
  title: string;
  image: string; // emoji stand-in for a dish photo
  description: string;
  cuisine: Cuisine;
  dietary: Dietary[]; // dietary guarantees this meal satisfies
  sitting: Sitting;
  neighborhood: string;
  price: number; // per seat, in GBP
  seatsTotal: number;
  /** Diner ids who have already joined this communal table. */
  seatsTaken: string[];
  communal: boolean; // true => open to meal buddies sharing the table
}

/** A fellow diner who can be matched as a meal buddy. */
export interface Diner {
  id: string;
  name: string;
  avatar: string;
  neighborhood: string;
  tastes: Cuisine[];
  dietary: Dietary[];
  bio: string;
}

/** What the current user tells the matcher they want tonight. */
export interface Preferences {
  cuisines: Cuisine[];
  dietary: Dietary[];
  neighborhood: string;
  sitting: Sitting | "any";
  budget: number; // max price per seat
  wantsBuddies: boolean;
}

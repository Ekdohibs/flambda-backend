(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)


(* What the meets of products keep: three meets put together, the meet
   of two maps, of the fields of two blocks, of two function types and
   of two entries of closures.

   This file is to be reviewed: the definitions of what it is for each
   of these parts to fit an environment and to be true, and the
   statements, which are about functions of FlambdaMeet.v. They are in
   the form of FlambdaMeetStepsSpec.v ([part_meet_ok]), and are for
   the proof of the contract of the whole meet (FlambdaMeetSpec.v).
   The proofs are in FlambdaMeetProductsProofs.v, which needs no
   review.

   What is stated.
   - [combine_results3] keeps what the three meets that it puts
     together keep ([combine_results3_ok_statement]), as
     [combine_results2] does for two.
   - The meet of two maps, [map_meet], is right if the meet of what
     they bind is ([map_meet_ok_statement]). A map fits if it binds a
     key once and what it binds fits; it is true if what it binds each
     key to is true, for that key. The map answered binds the keys
     that either input binds ([map_meet_keys_statement]).
   - What a meet answered, where its answer names an input
     ([meet_answer]): the three statements about keys, lengths and
     slots are about any such answer, and so about both inputs where
     the meet answers that both are the answer.
   - The meet of the fields of two blocks,
     [meet_int_indexed_product], is right
     ([meet_int_indexed_product_ok_statement]), and answers as many
     fields as the longer input has
     ([meet_int_indexed_product_length_statement]).
   - The meet of two function types is right
     ([meet_function_type_ok_statement]).
   - The meet of two entries of closures is right
     ([meet_closures_entry_ok_statement]), and the entry answered has
     types for the slots that either input has types for
     ([meet_closures_entry_slots_statement]).

   Two things that the statements about types say, and that a reader
   may not expect.

   What a part is true of. The meaning of types
   (FlambdaTypeMeaningSpec.v) asks nothing of a type that a product
   has for something that the value does not have: the type of a field
   beyond the fields of the block ([fields_satisfied]), the type of a
   function slot or of a value slot that the set of closures has not
   ([closures_entry_satisfied]). It is the index of the case that says
   what the value has, and the well-formedness of the case that keeps
   the product within the index (FlambdaTypeWellFormedSpec.v,
   conditions 6 and 8). The meet of two products meets the two types
   of a position whether or not the value has that position, and
   answers bottom if their meet is. So a meet of products loses no
   value only of the values that have every position of both
   products, and that is what "true" is here: [product_satisfied] and
   [closures_entry_holds] are the meaning of the product together with
   that. Where a product is met inside a case, the index and the
   well-formedness of the case give it for the fields of a block, and
   for the function types and the value slots of an entry of closures.
   For the closure types of an entry they do not give it alone. A
   closure that is bound to a symbol is projected to another function
   slot of its set only if the static set names a symbol for that slot
   ([project_function_slot], FlambdaValueSpec.v), though the slot is
   among the functions of the set ([closure_in_set]). That every
   static set names a symbol for each of its function slots is part of
   what it is for a static environment to be valid ([statics_valid]
   and [project_function_slot_valid_statement],
   FlambdaClosuresSpec.v). On a static environment that is not valid
   the meet of two entries loses a value: one that has both entries
   because the meaning asks nothing of their closure types at such a
   slot, where those two types have no value in common. So the half
   of what a call of a meet is to keep that says that no value is lost
   supposes a valid static environment ([meet_call_ok],
   FlambdaMeetStepsSpec.v), and whoever uses
   [meet_closures_entry_ok_statement] for the rows of closures has
   that as well as the index and the well-formedness of the case.

   What it is for an entry of closures to fit. The well-formedness of
   an entry is relative to the index of its case
   ([closures_entry_well_formed]), and the two entries that are met
   are under two indices, the entry answered under a third. What is
   kept by the meet of entries is what does not depend on the index:
   an entry fits if it is well formed under some index. That the entry
   answered is within the index answered is for the meet of the cases
   to show, from [meet_closures_entry_slots_statement].

   What the statements suppose of [Meet_env.meet_type], the meet of
   two types, through the meet of expanded types that it is given:
   [meet_expanded_head_keeps] and [meet_expanded_head_sound]
   (FlambdaMeetSpec.v), as [meet_type_ok_statement]
   (FlambdaMeetEnvSpec.v) does. The types of the closures of a set
   are to be of kind value (condition 5 of the well-formedness of
   types), and stay so: the type that [Meet_env.meet_type] answers is
   of the kind of its inputs ([meet_type_keeps_statement]). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvKindedSpec FlambdaMeetGrammar
  FlambdaExpandHead FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec
  FlambdaMeetStepsSpec FlambdaMeetEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* Three meets, one after the other                                         *)
(* ======================================================================== *)

(* [combine_results3]: the meet of two things that are each made of
   three parts, by the meets of the parts, one after the other. The
   thing made of three parts that fit must fit, and it must be true
   exactly when its parts are. *)
Definition combine_results3_ok_statement : Prop :=
  ∀ (A B C D : Type)
    (fits_a : Typing_env.t -> A -> Prop)
    (satisfied_a : static_env -> env -> A -> Prop)
    (fits_b : Typing_env.t -> B -> Prop)
    (satisfied_b : static_env -> env -> B -> Prop)
    (fits_c : Typing_env.t -> C -> Prop)
    (satisfied_c : static_env -> env -> C -> Prop)
    (fits_d : Typing_env.t -> D -> Prop)
    (satisfied_d : static_env -> env -> D -> Prop)
    (meet_a : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (meet_c : Meet_env.t -> C -> C -> option (Meet_result.t C))
    (rebuild : A -> B -> C -> D) (menv : Meet_env.t) (left_a right_a : A)
    (left_b right_b : B) (left_c right_c : C) (result : Meet_result.t D),
    let tenv := Meet_env.typing_env menv in
    part_stable fits_a satisfied_a ->
    part_stable fits_b satisfied_b ->
    part_stable fits_c satisfied_c ->
    part_meet_ok fits_a satisfied_a meet_a ->
    part_meet_ok fits_b satisfied_b meet_b ->
    part_meet_ok fits_c satisfied_c meet_c ->
    (∀ (tenv : Typing_env.t) (a : A) (b : B) (c : C),
       fits_a tenv a -> fits_b tenv b -> fits_c tenv c ->
       fits_d tenv (rebuild a b c)) ->
    (∀ (statics : static_env) (valuation : env) (a : A) (b : B) (c : C),
       satisfied_d statics valuation (rebuild a b c)
       <-> satisfied_a statics valuation a
           ∧ satisfied_b statics valuation b
           ∧ satisfied_c statics valuation c) ->
    typing_env_fits tenv ->
    fits_a tenv left_a ->
    fits_a tenv right_a ->
    fits_b tenv left_b ->
    fits_b tenv right_b ->
    fits_c tenv left_c ->
    fits_c tenv right_c ->
    combine_results3 menv meet_a meet_b meet_c left_a right_a left_b right_b
      left_c right_c rebuild
    = Some result ->
    meet_call_ok fits_d satisfied_d menv
      (rebuild left_a left_b left_c) (rebuild right_a right_b right_c) result.

(* ======================================================================== *)
(* What a meet answered                                                     *)
(* ======================================================================== *)

(* What a meet that answers [r] may be taken to have answered: the input
   that [r] names, either input if it names both, or the new thing.
   [extract_value] takes the first input where both are named. A meet
   of several parts that names its second input as the answer names it
   whole, though a part of it may have named both. *)
Definition meet_answer {A : Type} (r : meet_return_value A)
    (left right answer : A) : Prop :=
  match r with
  | Left_input => answer = left
  | Right_input => answer = right
  | Both_inputs => answer = left ∨ answer = right
  | New_result a => answer = a
  end.

Definition extract_value_answer_statement : Prop :=
  ∀ (A : Type) (r : meet_return_value A) (left right : A),
    meet_answer r left right (extract_value r left right).

(* ======================================================================== *)
(* Maps                                                                     *)
(* ======================================================================== *)

(* A map fits an environment: it binds a key once, and what it binds
   fits. *)
Definition map_fits {K A : Type} (eqb : K -> K -> bool)
    (fits : Typing_env.t -> A -> Prop) (tenv : Typing_env.t)
    (bindings : Keyed.finite_map K A) : Prop :=
  Keyed.distinct eqb (keys bindings) = true
  ∧ ∀ (key : K) (a : A),
      Keyed.lookup eqb key bindings = Some a -> fits tenv a.

(* A map is true: what it binds each key to is true, for that key. *)
Definition map_satisfied {K A : Type} (eqb : K -> K -> bool)
    (satisfied : K -> static_env -> env -> A -> Prop) (statics : static_env)
    (valuation : env) (bindings : Keyed.finite_map K A) : Prop :=
  ∀ (key : K) (a : A),
    Keyed.lookup eqb key bindings = Some a ->
    satisfied key statics valuation a.

(* [Map_meet.meet]: the meet of two maps is right, if the meet of what
   they bind is, for every key. *)
Definition map_meet_ok_statement : Prop :=
  ∀ (K A : Type) (eqb : K -> K -> bool)
    (fits : Typing_env.t -> A -> Prop)
    (satisfied : K -> static_env -> env -> A -> Prop)
    (meet_data : Meet_env.t -> A -> A -> option (Meet_result.t A)),
    decides_equality eqb ->
    (∀ key : K, part_stable fits (satisfied key)) ->
    (∀ key : K, part_meet_ok fits (satisfied key) meet_data) ->
    part_meet_ok (map_fits eqb fits) (map_satisfied eqb satisfied)
      (map_meet eqb meet_data).

(* The map answered binds the keys that either input binds. *)
Definition map_meet_keys_statement : Prop :=
  ∀ (K A : Type) (eqb : K -> K -> bool)
    (meet_data : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (menv menv' : Meet_env.t) (left right answer : Keyed.finite_map K A)
    (r : meet_return_value (Keyed.finite_map K A)) (key : K),
    decides_equality eqb ->
    map_meet eqb meet_data menv left right = Some (Meet_result.Ok r menv') ->
    meet_answer r left right answer ->
    Keyed.is_bound eqb key answer
    = Keyed.is_bound eqb key left || Keyed.is_bound eqb key right.

(* ======================================================================== *)
(* The fields of a block                                                    *)
(* ======================================================================== *)

(* The types of the fields of a block fit an environment: each does. *)
Definition product_fits (tenv : Typing_env.t) (tys : list flambda_type)
    : Prop :=
  ∀ ty : flambda_type, ty ∈ tys -> type_fits tenv ty.

(* They are true of the fields of a block: the block has a field for
   each, which has the type. This is [fields_satisfied], and that the
   types are no more than the fields
   ([product_satisfied_fields_statement]). *)
Definition product_satisfied (fields : list value) (statics : static_env)
    (valuation : env) (tys : list flambda_type) : Prop :=
  ∀ (i : nat) (ty : flambda_type),
    nth_error tys i = Some ty ->
    ∃ field : value,
      nth_error fields i = Some field
      ∧ type_satisfied statics valuation field ty.

Definition product_satisfied_fields_statement : Prop :=
  ∀ (fields : list value) (statics : static_env) (valuation : env)
    (tys : list flambda_type),
    product_satisfied fields statics valuation tys
    <-> length tys <= length fields
        ∧ fields_satisfied (type_satisfied statics valuation) tys fields.

(* [meet_int_indexed_product]: the meet of the types of the fields of
   two blocks is right, for the fields of any block. *)
Definition meet_int_indexed_product_ok_statement : Prop :=
  ∀ (mh : meet_expanded_head) (fields : list value),
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    part_meet_ok product_fits (product_satisfied fields)
      (meet_int_indexed_product mh).

(* It answers as many types as the longer input has. *)
Definition meet_int_indexed_product_length_statement : Prop :=
  ∀ (mh : meet_expanded_head) (menv menv' : Meet_env.t)
    (fields1 fields2 answer : list flambda_type)
    (r : meet_return_value (list flambda_type)),
    meet_int_indexed_product mh menv fields1 fields2
    = Some (Meet_result.Ok r menv') ->
    meet_answer r fields1 fields2 answer ->
    length answer = Nat.max (length fields1) (length fields2).

(* ======================================================================== *)
(* Function types                                                           *)
(* ======================================================================== *)

(* The type of the function at a slot is true of a set of closures: the
   set has a function at the slot, whose code is the one that the type
   names, if it names one. *)
(* CR mshinwell: the meet of two function types that name two codes is
   bottom here. The compiler asks the code age relation, and answers
   the newer of the two if one is a newer version of the other
   ([meet_code_id]); the meaning of a function type here is the very
   code of the closure, with no newer version. So nothing is claimed
   of the compiler's meet of the function types of two versions of one
   code. Discrepancy T4-4. *)
Definition function_type_satisfied
    (functions : list (function_slot * code_id)) (slot : function_slot)
    (_ : static_env) (_ : env) (function_type : Or_unknown.t function_type)
    : Prop :=
  ∃ code : code_id,
    lookup slot functions = Some code
    ∧ match function_type with
      | Or_unknown.Known function_type =>
        code = function_type_code_id function_type
      | Or_unknown.Unknown => True
      end.

(* [meet_function_type] is right. Every function type fits. *)
Definition meet_function_type_ok_statement : Prop :=
  ∀ (functions : list (function_slot * code_id)) (slot : function_slot),
    part_meet_ok
      (fun (_ : Typing_env.t) (_ : Or_unknown.t function_type) => True)
      (function_type_satisfied functions slot) meet_function_type.

(* ======================================================================== *)
(* Entries of closures                                                      *)
(* ======================================================================== *)

(* An entry of closures fits an environment: it is well formed under
   some index, the environment knows its variables, and it is kinded
   there, the names of the equations of its extensions included. *)
Definition closures_entry_fits (tenv : Typing_env.t) (entry : closures_entry)
    : Prop :=
  (∃ index : row_like_index set_of_closures_contents unit,
     closures_entry_well_formed index entry = true)
  ∧ (∀ x : variable,
      x ∈ free_variables_of_closures_entry entry ->
      Typing_env.mem None tenv (Name_var x) = true)
  ∧ (∀ (s : simple) (kind : flambda_kind),
      (s, kind) ∈ aliases_of_closures_entry entry ->
      simple_of_kind tenv s kind)
  ∧ ∀ (n : name) (kind : flambda_kind),
      (n, kind) ∈ equation_names_of_closures_entry entry ->
      simple_of_kind tenv (simple_of_name n) kind.

(* The closure [v], at [slot] of a set of closures with [functions] and
   [value_slots], has every slot that the entry has a type for. *)
Definition closures_entry_slots_present (statics : static_env) (v : value)
    (slot : function_slot) (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) (entry : closures_entry)
    : Prop :=
  match entry with
  | Closures_entry function_types closure_types value_slot_types =>
    (∀ g : function_slot,
       is_bound g function_types = true -> is_bound g functions = true)
    ∧ (∀ g : function_slot,
         is_bound g closure_types = true ->
         ∃ closure : value,
           project_function_slot statics slot g v = Some closure)
    ∧ ∀ w : value_slot,
        is_bound w value_slot_types = true -> is_bound w value_slots = true
  end.

(* An entry is true of the closure: [closures_entry_satisfied], and the
   closure has the slots of the entry. *)
Definition closures_entry_holds (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) (statics : static_env)
    (valuation : env) (entry : closures_entry) : Prop :=
  closures_entry_satisfied statics valuation v slot functions value_slots
    entry
  ∧ closures_entry_slots_present statics v slot functions value_slots entry.

(* [meet_closures_entry] is right, for any closure. *)
Definition meet_closures_entry_ok_statement : Prop :=
  ∀ (mh : meet_expanded_head) (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)),
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    part_meet_ok closures_entry_fits
      (closures_entry_holds v slot functions value_slots)
      (meet_closures_entry mh).

(* The entry answered has a type for a slot if one of the inputs
   has. *)
Definition meet_closures_entry_slots_statement : Prop :=
  ∀ (mh : meet_expanded_head) (menv menv' : Meet_env.t)
    (entry1 entry2 answer : closures_entry)
    (r : meet_return_value closures_entry),
    meet_closures_entry mh menv entry1 entry2
    = Some (Meet_result.Ok r menv') ->
    meet_answer r entry1 entry2 answer ->
    match entry1, entry2, answer with
    | Closures_entry function_types1 closure_types1 value_slot_types1,
      Closures_entry function_types2 closure_types2 value_slot_types2,
      Closures_entry function_types closure_types value_slot_types =>
      (∀ g : function_slot,
         is_bound g function_types
         = is_bound g function_types1 || is_bound g function_types2)
      ∧ (∀ g : function_slot,
           is_bound g closure_types
           = is_bound g closure_types1 || is_bound g closure_types2)
      ∧ ∀ w : value_slot,
          is_bound w value_slot_types
          = is_bound w value_slot_types1 || is_bound w value_slot_types2
    end.

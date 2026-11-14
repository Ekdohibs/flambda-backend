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

(* The locations of the heap as a sort of names: the action of their
   permutations on what the machine holds.

   This file is to be reviewed: the action ([permute_value] and the
   functions after it), the locations of each thing, and the
   statements.

   A location is the name of a cell of the heap (FlambdaValueSpec.v).
   It is compared by equality only, and nothing is meant to depend on
   which number it is: a run that allocates chooses a location that
   the heap does not bind, and from a state all of whose locations
   its heap binds another choice should give the same run with other
   names. That is the property this file prepares for; it is not
   stated here ("What is not here" says where it is). So the locations
   are treated as a sort of names, the third beside the variables and
   the continuations, and what is said of names in
   FlambdaNameAbstractionSpec.v is said of them here.

   Where locations are. A value may be a location or hold one in a
   field, a value slot or an argument kept. So locations are in the
   environments of variables, in the continuations (the environment of
   a closure, the values kept with a continuation), in the statically
   allocated values, in the heap, whose keys are locations and whose
   cells hold values, and in the read record
   (FlambdaValueSpec.read_record), whose entries are at locations.
   There is none in a term, in code, in a constant, in a kind or in a
   primitive, and there is no binder of locations in a term: the only
   thing that makes one is an allocation, in the machine.

   The action. A permutation of locations
   (FlambdaNominalSpec.permutation, a list of swaps) acts on values
   ([permute_value]: a location is moved, and the values inside a
   value are gone into), on cells and heaps ([permute_cell],
   [permute_heap]: the keys of the heap are moved with the values of
   its cells; the mutability of a block, its tag and the kind of an
   array are as they were), on read records ([permute_reads]: the
   location of an entry is moved, and what was read there is as it
   was), on environments, continuations and static environments
   ([permute_env], [permute_cont_value], [permute_static_env]:
   handlers and code are left as they are), and on states, on what a
   step comes to and on behaviours ([permute_state],
   [permute_result], [permute_behaviour]: the expression and the
   continuation of an exit are left as they are).

   The locations of a thing ([locations_of_cell] and the functions
   after it; those of a value are FlambdaHeapSpec.locations_of_value)
   are all the locations that occur in it, the keys of a heap among
   them. They are lists and a location may be there more than once.

   What is stated.
   - Each of the seven kinds of thing, with its action, its locations
     and the equality [=], obeys the laws of
     FlambdaNameAbstractionSpec.nominal: the identity does nothing,
     the action of a composition is the composition of the actions,
     two lists of swaps that are the same permutation act alike, the
     locations of a thing moved are its locations moved, and a
     permutation that fixes the locations of a thing leaves it as it
     is. So each is a nominal set whose support is its locations. A
     read record has no statement of its own: it is a component of a
     state.
   - The action on states has inverses
     ([permute_state_inverse_statement]).
   - This action and that of the renamings of variables and
     continuations (FlambdaEquivarianceSpec.rename_state) commute
     ([rename_permute_state_statement]): the first moves nothing in a
     term and the second nothing in a value.
   - Looking a location up in a heap, and a variable up in an
     environment, go through the action
     ([lookup_permute_heap_statement],
     [lookup_permute_env_statement]); the kind of a value is that of
     the value moved ([kind_of_permute_value_statement]); and a heap
     moved binds a location moved exactly where the heap binds the
     location ([is_bound_permute_heap_statement]).

   What is not here. That the machine goes through the action: that
   is in two other files. A step that allocates chooses a location
   outside the domain of the heap, computed from the locations that
   the heap binds (FlambdaHeapSpec.fresh_location), and that choice
   does not commute with a permutation, as no choice of a name does
   (NAMES.md, section 7). So the statements there are about a heap
   that binds every location at hand, those of an entry of the heap
   that a later write hides among them (a location that is held and
   is not bound may be the one that an allocation chooses, and then
   the allocation captures it), and they are up to a second
   permutation that agrees with the first on the locations that the
   heap binds: FlambdaLocationsPrimitiveSpec.v for the primitives,
   the two that allocate among them, and
   FlambdaLocationsMachineSpec.v for a step, a run and a behaviour
   of the machine, whose [Let] runs those primitives. The link with
   the relation between outcomes of FlambdaHeapMachineSpec.v, which pairs
   locations by a correspondence, is not stated.

   Nothing is trusted here. The proofs are in
   FlambdaLocationsProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNameAbstractionSpec
  FlambdaEquivarianceSpec FlambdaHeapSpec.
Import ListNotations.

(* ======================================================================== *)
(* The action                                                               *)
(* ======================================================================== *)

(* A value: a location is moved, and the values that a value holds are
   gone into. *)
Fixpoint permute_value (p : permutation) (v : value) : value :=
  match v with
  | Value_const c => Value_const c
  | Value_closure code slots =>
    Value_closure code
      (map (fun slot => (fst slot, permute_value p (snd slot))) slots)
  | Value_closure_of_set slot decls slots =>
    Value_closure_of_set slot decls
      (map (fun slot => (fst slot, permute_value p (snd slot))) slots)
  | Value_symbol s => Value_symbol s
  | Value_block tag fields => Value_block tag (map (permute_value p) fields)
  | Value_immutable_array elements =>
    Value_immutable_array (map (permute_value p) elements)
  | Value_boxed_number c => Value_boxed_number c
  | Value_location l => Value_location (permute p l)
  | Value_partial_application callee args args_arity =>
    Value_partial_application (permute_value p callee)
      (map (permute_value p) args) args_arity
  end.

(* A cell: its values. The mutability and the tag of a block and the
   kind of an array are as they were. *)
Definition permute_cell (p : permutation) (c : cell) : cell :=
  match c with
  | Cell_block mut tag fields =>
    Cell_block mut tag (map (permute_value p) fields)
  | Cell_array kind elements =>
    Cell_array kind (map (permute_value p) elements)
  end.

(* A heap: its keys, and the values of its cells. *)
Definition permute_heap (p : permutation) (h : heap) : heap :=
  map (fun binding => (permute p (fst binding), permute_cell p (snd binding)))
    h.

(* The read record: the locations of its entries. *)
Definition permute_reads (p : permutation) (rd : read_record) : read_record :=
  map (fun entry => (permute p (fst entry), snd entry)) rd.

(* An environment of variables: its values. *)
Definition permute_env (p : permutation) (vars : env) : env :=
  map_values (permute_value p) vars.

(* A statically allocated value: its values. *)
Definition permute_static_value (p : permutation)
    (a : statically_allocated_value) : statically_allocated_value :=
  match a with
  | Static_closure code slots =>
    Static_closure code
      (map (fun slot => (fst slot, permute_value p (snd slot))) slots)
  | Static_closure_of_set slot closure_symbols decls slots =>
    Static_closure_of_set slot closure_symbols decls
      (map (fun slot => (fst slot, permute_value p (snd slot))) slots)
  | Static_block tag fields => Static_block tag (map (permute_value p) fields)
  | Static_boxed_number contents =>
    Static_boxed_number (permute_value p contents)
  | Static_string bytes => Static_string bytes
  | Static_immutable_array elements =>
    Static_immutable_array (map (permute_value p) elements)
  | Static_imported c => Static_imported c
  end.

(* The static environment: its statically allocated values. Code holds
   no location. *)
Definition permute_static_env (p : permutation) (statics : static_env)
    : static_env :=
  Static_env (static_code statics)
    (map_values (permute_static_value p) (static_values statics)).

(* A continuation: the environment of a closure, the continuations that
   it holds, and the values kept. Handlers hold no location. *)
Fixpoint permute_cont_value (p : permutation) (c : cont_value) : cont_value :=
  match c with
  | Closure h vars conts recursive =>
    Closure h (permute_env p vars)
      (map (fun binding => (fst binding, permute_cont_value p (snd binding)))
         conts)
      recursive
  | Outer k => Outer k
  | Extra_args c extra =>
    Extra_args (permute_cont_value p c) (map (permute_value p) extra)
  | Over_application remaining remaining_arity return_arity return_to exn_to =>
    Over_application (map (permute_value p) remaining) remaining_arity
      return_arity (permute_cont_value p return_to)
      (permute_cont_value p exn_to)
  end.

Definition permute_cont_env (p : permutation) (conts : cont_env) : cont_env :=
  map (fun binding => (fst binding, permute_cont_value p (snd binding))) conts.

(* A state: everything but its expression. *)
Definition permute_state (p : permutation) (s : state) : state :=
  State (state_expr s) (permute_env p (state_vars s))
    (permute_cont_env p (state_conts s))
    (permute_static_env p (state_statics s)) (permute_heap p (state_heap s))
      (permute_reads p (state_reads s)).

(* What a step comes to, and a behaviour: the values and the heap of an
   exit, and the read record of what a step comes to (a behaviour has
   none). The continuation of an exit is not a location. *)
Definition permute_result (p : permutation) (result : step_result)
    : step_result :=
  match result with
  | Step s => Step (permute_state p s)
  | Exit k vs hp rd => Exit k (map (permute_value p) vs) (permute_heap p hp)
    (permute_reads p rd)
  | Undefined => Undefined
  end.

Definition permute_behaviour (p : permutation) (b : behaviour) : behaviour :=
  match b with
  | Exits k vs hp => Exits k (map (permute_value p) vs) (permute_heap p hp)
  | Goes_wrong => Goes_wrong
  | Diverges => Diverges
  end.

(* ======================================================================== *)
(* The locations of a thing                                                 *)
(* ======================================================================== *)

(* Those of a value are FlambdaHeapSpec.locations_of_value. *)

Definition locations_of_cell (c : cell) : list location :=
  flat_map locations_of_value (values_of_cell c).

(* The keys of a heap are among its locations. *)
Definition locations_of_heap (h : heap) : list location :=
  flat_map (fun binding => fst binding :: locations_of_cell (snd binding)) h.

Definition locations_of_reads (rd : read_record) : list location :=
  map fst rd.

Definition locations_of_env (vars : env) : list location :=
  flat_map (fun binding => locations_of_value (snd binding)) vars.

Definition locations_of_static_value (a : statically_allocated_value)
    : list location :=
  flat_map locations_of_value (values_of_static a).

Definition locations_of_static_env (statics : static_env) : list location :=
  flat_map (fun binding => locations_of_static_value (snd binding))
    (static_values statics).

Fixpoint locations_of_cont_value (c : cont_value) : list location :=
  match c with
  | Closure _ vars conts _ =>
    locations_of_env vars
    ++ flat_map (fun binding => locations_of_cont_value (snd binding)) conts
  | Outer _ => []
  | Extra_args c extra =>
    locations_of_cont_value c ++ flat_map locations_of_value extra
  | Over_application remaining _ _ return_to exn_to =>
    flat_map locations_of_value remaining
    ++ locations_of_cont_value return_to ++ locations_of_cont_value exn_to
  end.

Definition locations_of_cont_env (conts : cont_env) : list location :=
  flat_map (fun binding => locations_of_cont_value (snd binding)) conts.

Definition locations_of_state (s : state) : list location :=
  locations_of_env (state_vars s)
  ++ locations_of_cont_env (state_conts s)
  ++ locations_of_static_env (state_statics s)
  ++ locations_of_heap (state_heap s)
  ++ locations_of_reads (state_reads s).

(* ======================================================================== *)
(* Each is a nominal set                                                    *)
(* ======================================================================== *)

(* The things that hold locations, as FlambdaNameAbstractionSpec.v has
   the things that hold names: the action, the locations, and the
   equality, which is [=]. *)
Definition value_locations : Nominal.contains_names value :=
  Nominal.Contains_names permute_value locations_of_value eq.

Definition cell_locations : Nominal.contains_names cell :=
  Nominal.Contains_names permute_cell locations_of_cell eq.

Definition heap_locations : Nominal.contains_names heap :=
  Nominal.Contains_names permute_heap locations_of_heap eq.

Definition env_locations : Nominal.contains_names env :=
  Nominal.Contains_names permute_env locations_of_env eq.

Definition static_env_locations : Nominal.contains_names static_env :=
  Nominal.Contains_names permute_static_env locations_of_static_env eq.

Definition cont_value_locations : Nominal.contains_names cont_value :=
  Nominal.Contains_names permute_cont_value locations_of_cont_value eq.

Definition state_locations : Nominal.contains_names state :=
  Nominal.Contains_names permute_state locations_of_state eq.

(* The laws: the identity, composition, permutations that are the same,
   the locations of a thing moved, and that a permutation which fixes
   the locations of a thing leaves it as it is. *)
Definition value_locations_nominal_statement : Prop :=
  Nominal.nominal value_locations.

Definition cell_locations_nominal_statement : Prop :=
  Nominal.nominal cell_locations.

Definition heap_locations_nominal_statement : Prop :=
  Nominal.nominal heap_locations.

Definition env_locations_nominal_statement : Prop :=
  Nominal.nominal env_locations.

Definition static_env_locations_nominal_statement : Prop :=
  Nominal.nominal static_env_locations.

Definition cont_value_locations_nominal_statement : Prop :=
  Nominal.nominal cont_value_locations.

Definition state_locations_nominal_statement : Prop :=
  Nominal.nominal state_locations.

(* The action on states has inverses. *)
Definition permute_state_inverse_statement : Prop :=
  ∀ (p : permutation) (s : state),
    permute_state (inverse_permutation p) (permute_state p s) = s
    ∧ permute_state p (permute_state (inverse_permutation p) s) = s.

(* ======================================================================== *)
(* The two actions                                                          *)
(* ======================================================================== *)

(* A renaming of variables and continuations and a permutation of
   locations commute: the first moves nothing in a value, the second
   nothing in a term. *)
Definition rename_permute_state_statement : Prop :=
  ∀ (r : renaming) (p : permutation) (s : state),
    rename_state r (permute_state p s) = permute_state p (rename_state r s).

(* ======================================================================== *)
(* Reading through the action                                               *)
(* ======================================================================== *)

(* A heap moved binds the location moved to the cell moved. *)
Definition lookup_permute_heap_statement : Prop :=
  ∀ (p : permutation) (h : heap) (l : location),
    lookup (permute p l) (permute_heap p h)
    = option_map (permute_cell p) (lookup l h).

Definition is_bound_permute_heap_statement : Prop :=
  ∀ (p : permutation) (h : heap) (l : location),
    is_bound (permute p l) (permute_heap p h) = is_bound l h.

(* An environment moved binds a variable to its value moved. *)
Definition lookup_permute_env_statement : Prop :=
  ∀ (p : permutation) (vars : env) (x : variable),
    lookup x (permute_env p vars)
    = option_map (permute_value p) (lookup x vars).

(* The kind of a value does not depend on its locations. *)
Definition kind_of_permute_value_statement : Prop :=
  ∀ (p : permutation) (v : value),
    kind_of_value (permute_value p v) = kind_of_value v.

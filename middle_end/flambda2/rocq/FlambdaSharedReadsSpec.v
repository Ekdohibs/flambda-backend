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

(* Two reads of one value that the unit put in two places of its static
   constants: the second is the variable of the first. A normalisation
   that refines, for a unit run on its own.

   This file is to be reviewed: [place_held] and [place_value], which
   say what a static constant holds at a place; [places_held], what is
   known; [group_holds] and [groups_hold], the check; [read_of] and
   [share_reads], the rewrite; [share_static_reads], which is extracted
   and run by the validation of middle_end/flambda2/tests/semantics/;
   [classes_agree], what a run keeps; and the statements, the last
   eight of which are the rewrite computed on small units.

   What Simplify does. A value that is computed when the unit runs is
   a variable where the unit binds its static constants. Where an
   exported function uses it, the closure of the function becomes a
   symbol bound to a closure one of whose value slots holds the
   variable; where a module exports it, the block of the module
   becomes a symbol bound to a block one of whose fields is the
   variable. In the code of a function such a symbol has the type of
   what it is bound to, and the variable is not in scope there. A read
   of the place, the projection of the value slot or the load of the
   field, cannot become the variable, and stays: Simplify gives the
   variable that the read binds the type of an alias of the variable
   that the place holds
   (simplify/simplify_unary_primitive.ml,
   [simplify_project_value_slot] and
   [simplify_immutable_block_load0], the case [Need_meet], by
   [Simplify_common.simplify_projection]). A later read of a place
   that holds the same variable, of the same symbol or of another,
   finds the variable of the first read, which is in scope and is an
   alias of what the place holds (the case [Known_result] of the same
   two functions), and becomes it. So where the code of a function
   reads a value of the module through its own closure and through
   the block of the module, or through the closures of two functions
   that both hold it, the term after has one read where the term
   before has two.

   What is here. A place of a static constant is a value slot of a
   closure or a field of a block ([static_place]). [places_held e]
   lists, for each group of static bindings of the unit [e], outside
   code, the places of its constants that hold a variable, each with
   a class: the first symbol that the group binds, and the variable.
   Two places of one class are places that one group gave one
   variable. A lifting gives each [Let] that it lifts a group of its
   own, and the rounds of the normalisation make one group of the
   groups that follow one another directly
   (FlambdaNormaliseSpec.merge_statics): that is how the closure of a
   function and the block of its module are of one group where the
   validation runs this pass. [share_reads] goes through the whole
   unit, in code and in handlers too, with the reads made so far
   ([reads_made]: a class, and the variable that a [Let] bound to a
   read of a place of that class). A read is a [Let] of
   [Project_value_slot] or
   [Project_value_slot_from] of a symbol, or of a [Block_load] that is
   not marked [Mutable] of a symbol ([read_of]). A read of a place
   whose class has been read becomes a [Let] of the variable of that
   read; the [Let] stays, and its variable is then an alias, which
   other passes follow (FlambdaAliasesSpec.follow_aliases).
   A read of a class that has not been read is kept and remembered. A
   read is forgotten under a binder of its variable. A handler starts
   with the reads made where it is defined, less those of its
   parameters; the body of code starts with none; the code of a stub
   is left as it is, as FlambdaKnownSlotsSpec.read_slots leaves it and
   for the reason given there. Nothing else is rewritten.

   Why it is right. The machine makes undefined the binding of a
   symbol that is bound (FlambdaSemanticsSpec.eval_static_bindings),
   and no step but that of a [Let_static] changes the static
   environment: everything that the static environment of a state
   binds, that of every later state of the run binds to the same
   thing (FlambdaWellFormedSpec.run_extends_statics_statement). A
   group is executed in one step, with one environment for its
   variables. So if every group that binds a symbol of a class binds
   all the symbols of the class, each to a constant that holds the
   variable of the class at its place, then in every state of the run
   two places of one class, where both hold a value, hold the same
   value ([classes_agree]). That of the groups is what [groups_hold]
   checks, of every group of the unit, in code too
   ([group_keeps_classes_statement],
   [classes_agree_kept_statement]). Nothing is guessed, and nothing is
   believed of what [places_held] reads: the statements are of any
   list that passes the check. The second read, where it has a value,
   has then the value of the first, which its variable still holds
   ([share_read_sound_statement]); where it has none the unit was
   undefined. So the unit rewritten refines the unit, run on its own
   ([share_static_reads_program_refines_statement]). It is not an
   equivalence: a read that has no value in the unit (a load whose
   access kind the block does not meet, or a projection that names a
   function slot at which the closure is not) has one once it is the
   variable of another.

   The check reads a constant as the machine does ([place_held]): the
   value slot of a closure is the first that the constant lists for
   the slot (FlambdaSemanticsSpec.eval_value_slots, and [lookup] where
   the slot is projected), so a constant that lists a slot twice holds
   at that slot what it lists first, and the second is no place
   ([share_static_reads_slot_twice_statement]). No term that the
   compiler builds lists a slot twice.

   The function slot that a projection names is not looked at: where
   the projection has a value, it is the value of the slot, whatever
   function slot the closure is at ([read_of_place_statement]).

   What it assumes of a write. The places are places of static
   constants, which nothing writes: in the model a static block has no
   mark of mutability, and a store has no value on a symbol
   (FlambdaHeapSpec.exec_heap_primitive, the case [Block_set], which
   has a value on a location bound to a block made [Mutable] and on
   nothing else: [store_to_symbol_undefined_statement], of a [Let] as
   the machine runs it, and [exec_prim_store_to_symbol_statement], of
   the primitive). So a unit that writes a
   field of a static block is undefined where it writes, and the
   refinement says nothing of it; the check does not look at stores,
   and answers of such a unit as of any other
   ([share_static_reads_store_statement]). The pass shares only loads
   that are not marked [Mutable]. If a symbol comes to be the address
   of a cell that can be written (requests/C4-heap.md, section 2.7),
   a field of a static block made [Mutable] may change after its
   group has run. The invariant is then of the places that nothing
   writes: the value slots of a static closure, and the fields of a
   static block that is not made [Mutable]. Of a cell at a location a
   load that is not marked [Mutable] has no value on a block made
   [Mutable] (FlambdaHeapSpec.load_allowed; decision 4 of that
   request). The pass stays as it is only if the same is said of a
   cell at a symbol. If a load that is not marked [Mutable] comes to
   have a value on a static block that can be written, the pass must
   ask that the block is not one, and its statement must not be what
   gives such a load a meaning (the shape of X5-2, DECISIONS.md).

   The same holds of a unit that imports symbols, run from the static
   environment of its imports
   ([share_static_reads_refines_imports_statement]), and nothing is
   asked of the imports: an imported symbol is bound to something of
   which the model knows no contents (FlambdaValueSpec.v,
   [Static_imported]), so no place of it holds a value, and two
   places of one class agree there for nothing. If an import comes to
   have contents (requests/C5-imports-with-contents.md), that
   statement is to ask that the imports bind no symbol of a class,
   as FlambdaKnownSymbolsSpec.fold_known_unit_refines_imports_statement
   asks of the symbols that it folds. The contents that a [.cmx] file
   gives are not in that environment: the test tool puts them in
   front of both terms as static bindings
   (middle_end/flambda2/tests/tools/flsemantics.ml, [make_prefix]), a
   group as any other to this pass, whose places hold constants and
   symbols and no variable. It holds too of the unit followed by
   something ([share_static_reads_refines_observed_statement]): what
   may follow a unit (FlambdaObserversSpec.observers) binds nothing
   static, so it passes the check, and it is the same term in the two
   runs.

   The pass puts a variable and no symbol, so a unit that passes row 4
   of requests/C0-well-formed.md passes it once rewritten
   ([share_static_reads_in_scope_statement]); the variable is bound by
   a [Let] around the read, so the unit rewritten has no free variable
   that the unit has not ([share_static_reads_keeps_scope_statement]);
   and it commutes with renaming, the variables of the classes being
   renamed with the term
   ([share_static_reads_equivariant_statement]).

   Two reads of one place are shared before this
   (FlambdaShareProjectionsSpec.share_known_projections has the second
   projection of one slot from one closure stand for the first, and
   FlambdaCseSpec.share_known_primitives the second load of one
   field); they are two reads of one class, and are shared here too,
   in code and in the handlers of recursive continuations as elsewhere.

   What is not here: two places that two groups gave one variable
   ([share_static_reads_two_groups_statement]), where something stays
   between the two groups that keeps them from being made one; the
   elements of a static array and the contents of a static boxed
   number that are variables; two value slots of a closure that a
   [Let] makes with one variable in both (the closure is no symbol);
   a place of a symbol that the unit imports; a read whose variable
   is not in scope where the second is made (in another handler, in
   the code of another function).

   Two pairs of units given to the validation that runs this pass,
   one that the pass makes validated and one that must be refused,
   are computed in FlambdaSharedReadsPairsSpec.v.

   The proofs are in FlambdaSharedReadsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaBindersSpec
  FlambdaNominalSpec FlambdaWellFormedSpec FlambdaStaticsInScopeSpec
  FlambdaUnusedStaticsSpec FlambdaHeapSpec FlambdaImportsSpec
  FlambdaObserversSpec.
Import ListNotations.

(* ======================================================================== *)
(* The places of a static constant                                          *)
(* ======================================================================== *)

(* A place of a static constant: a value slot of a closure, or a field
   of a block, the first being field 0. *)
Inductive static_place : Type :=
  | Place_slot (slot : value_slot)
  | Place_field (field : Z).

Definition same_place (p p' : static_place) : bool :=
  match p, p' with
  | Place_slot slot, Place_slot slot' => Pos.eqb slot slot'
  | Place_field field, Place_field field' => Z.eqb field field'
  | Place_slot _, Place_field _ | Place_field _, Place_slot _ => false
  end.

(* What a static constant holds at a place, read as the machine reads
   what the constant makes: of the value slots of a closure, the first
   that is listed for the slot; of the fields of a block, the one of
   that number. A constant that is no closure has no value slot, and
   one that is no block has no field. *)
Definition place_held (const : static_const) (p : static_place)
    : option simple :=
  match p, const with
  | Place_slot slot,
    (Static_const_set_of_closures _ value_slots
    | Static_const_closure_of_set _ _ _ value_slots) =>
    lookup slot value_slots
  | Place_field field, Static_const_block _ fields =>
    if (0 <=? field)%Z then nth_error fields (Z.to_nat field) else None
  | _, _ => None
  end.

(* What a statically allocated value holds at a place. *)
Definition place_value (allocated : statically_allocated_value)
    (p : static_place) : option value :=
  match p, allocated with
  | Place_slot slot,
    (Static_closure _ slot_values
    | Static_closure_of_set _ _ _ slot_values) =>
    lookup slot slot_values
  | Place_field field, Static_block _ fields =>
    if (0 <=? field)%Z then nth_error fields (Z.to_nat field) else None
  | _, _ => None
  end.

(* The places that a static constant lists: the value slots of a
   closure, the fields of a block. *)
Definition const_places (const : static_const) : list static_place :=
  match const with
  | Static_const_set_of_closures _ value_slots
  | Static_const_closure_of_set _ _ _ value_slots =>
    map (fun slot => Place_slot (fst slot)) value_slots
  | Static_const_block _ fields =>
    map (fun n => Place_field (Z.of_nat n)) (seq 0 (length fields))
  | Static_const_boxed_number _ | Static_const_immutable_string _
  | Static_const_immutable_value_array _ =>
    []
  end.

(* ======================================================================== *)
(* What is known                                                            *)
(* ======================================================================== *)

(* What two places hold in common: the first symbol that their group
   binds, and the variable that the group gave them both. *)
Definition place_class : Type := symbol * variable.

Definition same_class (c c' : place_class) : bool :=
  Pos.eqb (fst c) (fst c') && Pos.eqb (snd c) (snd c').

(* Places of symbols, each with its class. A place may be listed more
   than once; the first is used. *)
Definition held_places : Type := list (symbol * static_place * place_class).

(* The first symbol that a group binds. *)
Fixpoint first_symbol (bindings : list static_binding) : option symbol :=
  match bindings with
  | [] => None
  | Bind_symbol s _ :: _ => Some s
  | (Bind_code _ _ | Bind_deleted_code _) :: bindings => first_symbol bindings
  end.

(* A place of the constant of [s], if it holds a variable, with the
   class of that variable in the group whose first symbol is
   [first]. *)
Definition place_entry (first s : symbol) (const : static_const)
    (p : static_place) : held_places :=
  match place_held const p with
  | Some (Var x) => [(s, p, (first, x))]
  | Some (Symbol _ | Const _) | None => []
  end.

Definition binding_places (first : symbol) (binding : static_binding)
    : held_places :=
  match binding with
  | Bind_symbol s const =>
    flat_map (place_entry first s const) (const_places const)
  | Bind_code _ _ | Bind_deleted_code _ => []
  end.

(* The places of the constants of a group that hold a variable. *)
Definition group_places (bindings : list static_binding) : held_places :=
  match first_symbol bindings with
  | Some first => flat_map (binding_places first) bindings
  | None => []
  end.

(* The same of all the groups of an expression, outside code. *)
Fixpoint places_held (e : expr) : held_places :=
  match e with
  | Let _ _ _ body => places_held body
  | Let_static bindings body => group_places bindings ++ places_held body
  | Let_cont_non_recursive _ h body =>
    places_held_of_handler h ++ places_held body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => places_held_of_handler h) handlers
    ++ places_held body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with places_held_of_handler (h : continuation_handler) : held_places :=
  match h with
  | Handler _ _ handler => places_held handler
  end.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* The constant that a group binds a symbol to: the first, if it binds
   it more than once; the machine has no step for a group that does. *)
Fixpoint group_const (s : symbol) (bindings : list static_binding)
    : option static_const :=
  match bindings with
  | [] => None
  | Bind_symbol s' const :: bindings =>
    if Pos.eqb s s' then Some const else group_const s bindings
  | (Bind_code _ _ | Bind_deleted_code _) :: bindings =>
    group_const s bindings
  end.

(* Whether an entry has to do with a group: the group binds its symbol,
   or the first symbol that the group binds is that of its class. *)
Definition entry_meets_group (bindings : list static_binding)
    (entry : symbol * static_place * place_class) : bool :=
  let '(s, _, (first, _)) := entry in
  is_some (group_const s bindings)
  || match first_symbol bindings with
     | Some first' => Pos.eqb first first'
     | None => false
     end.

(* Whether an entry is of a group: the first symbol that the group
   binds is that of its class, and the group binds its symbol to a
   constant that holds the variable of its class at its place. *)
Definition entry_of_group (bindings : list static_binding)
    (entry : symbol * static_place * place_class) : bool :=
  let '(s, p, (first, x)) := entry in
  match first_symbol bindings, group_const s bindings with
  | Some first', Some const =>
    Pos.eqb first first'
    && match place_held const p with
       | Some (Var x') => Pos.eqb x x'
       | Some (Symbol _ | Const _) | None => false
       end
  | _, _ => false
  end.

(* Every entry that has to do with the group is of the group. So a
   group that binds a symbol of a class binds all the symbols of the
   class, each to a constant that holds the variable of the class at
   its place. *)
Definition group_holds (held : held_places) (bindings : list static_binding)
    : bool :=
  forallb
    (fun entry =>
       negb (entry_meets_group bindings entry)
       || entry_of_group bindings entry)
    held.

(* That of every group of the expression, in handlers and in code
   too. *)
Fixpoint groups_hold (held : held_places) (e : expr) : bool :=
  match e with
  | Let _ _ _ body => groups_hold held body
  | Let_static bindings body =>
    group_holds held bindings
    && forallb (code_of_binding_groups_hold held) bindings
    && groups_hold held body
  | Let_cont_non_recursive _ h body =>
    handler_groups_hold held h && groups_hold held body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_groups_hold held h) handlers
    && groups_hold held body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_groups_hold (held : held_places) (h : continuation_handler)
    : bool :=
  match h with
  | Handler _ _ handler => groups_hold held handler
  end
with code_of_binding_groups_hold (held : held_places)
    (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_groups_hold held c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_groups_hold (held : held_places) (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => groups_hold held body
  end.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* The place that a defining expression reads, and the symbol that it
   reads it of: the projection of a value slot from a symbol, whatever
   function slot it names, and the load of a field from a symbol, where
   the load is not marked [Mutable]. *)
Definition read_of (d : named) : option (symbol * static_place) :=
  match d with
  | Project_value_slot slot (Symbol s)
  | Project_value_slot_from _ slot (Symbol s) =>
    Some (s, Place_slot slot)
  | Prim (Unary (Block_load _ (Immutable | Immutable_unique) field))
      [Symbol s] =>
    Some (s, Place_field field)
  | _ => None
  end.

(* The class of a place of a symbol: that of the first entry for it. *)
Fixpoint class_of (held : held_places) (s : symbol) (p : static_place)
    : option place_class :=
  match held with
  | [] => None
  | (s', p', c) :: held =>
    if Pos.eqb s s' && same_place p p' then Some c else class_of held s p
  end.

(* The class of the place that a defining expression reads. *)
Definition read_class (held : held_places) (d : named) : option place_class :=
  match read_of d with
  | Some (s, p) => class_of held s p
  | None => None
  end.

(* The reads made: classes, each with the variable that a [Let] bound
   to a read of a place of that class. A class may be listed more than
   once; the first is used. *)
Definition reads_made : Type := list (place_class * variable).

(* The reads made but for those whose variable is [x]: once [x] is
   bound again, it stands for something else. *)
Definition forget_read (x : variable) (made : reads_made) : reads_made :=
  filter (fun entry => negb (Pos.eqb x (snd entry))) made.

(* The same of parameters, bound one after the other. *)
Fixpoint forget_reads (xs : list variable) (made : reads_made) : reads_made :=
  match xs with
  | [] => made
  | x :: xs => forget_reads xs (forget_read x made)
  end.

(* The variable of the first read made of a class. *)
Fixpoint read_before (made : reads_made) (c : place_class)
    : option variable :=
  match made with
  | [] => None
  | (c', v) :: made =>
    if same_class c c' then Some v else read_before made c
  end.

(* Every [Let] of the expression, in handlers and in code too, but for
   the code of a stub, which is left as it is. A read of a class that
   has been read becomes the variable of that read; a read of a class
   that has not is remembered for what follows. *)
Fixpoint share_reads (held : held_places) (made : reads_made) (e : expr)
    {struct e} : expr :=
  match e with
  | Let x kind d body =>
    match read_class held d with
    | Some c =>
      match read_before made c with
      | Some v =>
        Let x kind (Simple (Var v))
          (share_reads held (forget_read x made) body)
      | None =>
        Let x kind d (share_reads held ((c, x) :: forget_read x made) body)
      end
    | None => Let x kind d (share_reads held (forget_read x made) body)
    end
  | Let_static bindings body =>
    Let_static (map (share_reads_in_binding held) bindings)
      (share_reads held made body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (share_reads_in_handler held made h)
      (share_reads held made body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, share_reads_in_handler held made h)) handlers)
      (share_reads held made body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with share_reads_in_handler (held : held_places) (made : reads_made)
    (h : continuation_handler) {struct h} : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (share_reads held (forget_reads params made) handler)
  end
with share_reads_in_binding (held : held_places) (binding : static_binding)
    {struct binding} : static_binding :=
  match binding with
  | Bind_code id c =>
    if code_is_stub c then binding
    else Bind_code id (share_reads_in_code held c)
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end
with share_reads_in_code (held : held_places) (c : code) {struct c} : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    Code params my_closure ret exn metadata (share_reads held [] body)
  end.

(* The unit with the reads of the places that it is read to give one
   variable shared, if the check passes of what was read; the unit as
   it is otherwise. *)
Definition share_static_reads (e : expr) : expr :=
  let held := places_held e in
  if groups_hold held e then share_reads held [] e else e.

(* ======================================================================== *)
(* What a run keeps                                                         *)
(* ======================================================================== *)

(* Two places of one class, where the static environment binds their
   symbols and both hold a value, hold the same value. *)
Definition classes_agree (held : held_places) (statics : static_env) : Prop :=
  ∀ (s s' : symbol) (p p' : static_place) (c : place_class)
    (allocated allocated' : statically_allocated_value) (v v' : value),
    (s, p, c) ∈ held ->
    (s', p', c) ∈ held ->
    lookup s (static_values statics) = Some allocated ->
    lookup s' (static_values statics) = Some allocated' ->
    place_value allocated p = Some v ->
    place_value allocated' p' = Some v' ->
    v' = v.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* What a static constant makes holds, at each place, the value of what
   the constant holds there. *)
Definition place_held_value_statement : Prop :=
  ∀ (vars : env) (const : static_const)
    (allocated : statically_allocated_value) (p : static_place) (a : simple),
    eval_static_const vars const = Some allocated ->
    place_held const p = Some a ->
    place_value allocated p = eval_simple vars a.

(* A read that has a value has the value that the place holds, of what
   the symbol is bound to. *)
Definition read_of_place_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (d : named) (s : symbol)
    (p : static_place) (v : value),
    read_of d = Some (s, p) ->
    eval_named statics vars d = Some v ->
    ∃ allocated : statically_allocated_value,
      lookup s (static_values statics) = Some allocated
      ∧ place_value allocated p = Some v.

(* A read looks at no heap and changes none: where it has a value on a
   heap, it has that value by [eval_named], and the heap is as it
   came. *)
Definition read_of_exec_statement : Prop :=
  ∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record) (vars : env)
    (d : named)
    (s : symbol) (p : static_place) (v : value),
    read_of d = Some (s, p) ->
    exec_named statics hp rd vars d = Some (v, hp', rd') ->
    eval_named statics vars d = Some v ∧ hp' = hp.

(* A group that passes the check, executed in a static environment in
   which the places of each class agree, gives one in which they
   do. *)
Definition group_keeps_classes_statement : Prop :=
  ∀ (held : held_places) (vars : env) (bindings : list static_binding)
    (statics statics' : static_env),
    group_holds held bindings = true ->
    classes_agree held statics ->
    eval_static_bindings vars bindings statics = Some statics' ->
    classes_agree held statics'.

(* So they agree in every state of the run of a unit that passes the
   check, run on its own. *)
Definition classes_agree_kept_statement : Prop :=
  ∀ (held : held_places) (e : expr) (n : nat) (s : state),
    groups_hold held e = true ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    classes_agree held (state_statics s).

(* One read: where the places of each class agree, a read that has a
   value has the value that any place of its class holds. *)
Definition share_read_sound_statement : Prop :=
  ∀ (held : held_places) (statics : static_env) (vars : env) (d : named)
    (c : place_class) (s : symbol) (p : static_place)
    (allocated : statically_allocated_value) (v w : value),
    classes_agree held statics ->
    read_class held d = Some c ->
    (s, p, c) ∈ held ->
    lookup s (static_values statics) = Some allocated ->
    place_value allocated p = Some v ->
    eval_named statics vars d = Some w ->
    w = v.

(* The rewrite changes no static constant and makes no group: a term
   passes the check before it if and only if it does after. *)
Definition share_reads_groups_hold_statement : Prop :=
  ∀ (held held' : held_places) (made : reads_made) (e : expr),
    groups_hold held' (share_reads held made e) = groups_hold held' e.

(* A unit that passes the check, with its reads shared, refines the
   unit, run on its own. A refinement, and no equivalence. *)
Definition share_reads_program_refines_statement : Prop :=
  ∀ (held : held_places) (e : expr),
    groups_hold held e = true ->
    program_refines e (share_reads held [] e).

(* So what [share_static_reads] gives refines the unit, run on its
   own. *)
Definition share_static_reads_program_refines_statement : Prop :=
  ∀ e : expr, program_refines e (share_static_reads e).

(* The same from a static environment that binds something already:
   the code that it has passes the check, and the places of each class
   agree in it. The heap is empty. *)
Definition share_reads_refines_from_statement : Prop :=
  ∀ (held : held_places) (e : expr) (statics : static_env),
    groups_hold held e = true ->
    Forall (fun binding => code_groups_hold held (snd binding) = true)
      (static_code statics) ->
    classes_agree held statics ->
    refines (State e [] [] statics empty_heap no_reads)
      (State (share_reads held [] e) [] [] statics empty_heap no_reads).

(* So what [share_static_reads] gives refines the unit, each run on its
   own from the same imports (FlambdaImportsSpec.unit_refines): the
   static environment of the imports has no code, and binds each
   symbol to something imported, no place of which holds a value. *)
Definition share_static_reads_refines_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (e : expr),
    unit_refines imports e (share_static_reads e).

(* The same of the unit followed by something: the unit and the unit
   rewritten, run from one continuation environment of what may follow
   a unit (FlambdaObserversSpec.observers: a way out, or a handler with
   no environment of its own that binds nothing static and names no
   symbol and no code id). What follows is the same term in the two
   runs: it is not rewritten. It binds nothing static, so it passes
   the check and the places of each class still agree when it runs.
   With no continuation this is [share_reads_refines_from_statement]. *)
Definition share_reads_refines_observed_statement : Prop :=
  ∀ (held : held_places) (e : expr) (conts : cont_env)
    (statics : static_env),
    groups_hold held e = true ->
    Forall (fun binding => code_groups_hold held (snd binding) = true)
      (static_code statics) ->
    classes_agree held statics ->
    observers [] conts ->
    refines (State e [] conts statics empty_heap no_reads)
      (State (share_reads held [] e) [] conts statics empty_heap no_reads).

(* So what [share_static_reads] gives refines the unit, the two
   followed by the same thing and run from the same imports, with the
   same exits. *)
Definition share_static_reads_refines_observed_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (conts : cont_env) (e : expr),
    observers [] conts ->
    refines (State e [] conts (imports_statics imports) empty_heap no_reads)
      (State (share_static_reads e) [] conts (imports_statics imports)
         empty_heap no_reads).

(* The rewrite keeps row 4: it puts a variable, and no symbol. Where
   every symbol and code id that a term mentions is in scope, given
   the symbols [syms] and the code ids [ids] in scope around it, the
   same holds of the term rewritten. *)
Definition share_reads_in_scope_statement : Prop :=
  ∀ (held : held_places) (made : reads_made) (syms : list symbol)
    (ids : list code_id) (e : expr),
    expr_statics_in_scope syms ids e = true ->
    expr_statics_in_scope syms ids (share_reads held made e) = true.

Definition share_static_reads_in_scope_statement : Prop :=
  ∀ (imports : list symbol) (e : expr),
    statics_in_scope imports e = true ->
    statics_in_scope imports (share_static_reads e) = true.

(* The variable that the rewrite puts is bound by a [Let] around the
   read. On a term whose code is closed, the code of the result is
   closed, and a variable or a continuation that is free in the result
   is free in the term: this is
   FlambdaValidatorEquivarianceSpec.keeps_scope of the rewrite, written
   out, as that file comes after this one. *)
Definition share_static_reads_keeps_scope_statement : Prop :=
  ∀ e : expr,
    code_closed e = true ->
    code_closed (share_static_reads e) = true
    ∧ (∀ x : variable,
        among x (free_variables_of_expr (share_static_reads e)) = true ->
        among x (free_variables_of_expr e) = true)
    ∧ (∀ k : continuation,
        among k (free_continuations_of_expr (share_static_reads e)) = true ->
        among k (free_continuations_of_expr e) = true).

(* The action of a renaming on what is known and on the reads made:
   the variables are renamed, the symbols and the places are not. *)
Definition rename_held (r : renaming) (held : held_places) : held_places :=
  map
    (fun '(s, p, (first, x)) => (s, p, (first, rename_variable r x)))
    held.

Definition rename_reads_made (r : renaming) (made : reads_made)
    : reads_made :=
  map
    (fun '((first, x), v) =>
       ((first, rename_variable r x), rename_variable r v))
    made.

(* The rewrite commutes with renaming, what is known and the reads
   made being renamed with the term. *)
Definition share_reads_equivariant_statement : Prop :=
  ∀ (r : renaming) (held : held_places) (made : reads_made) (e : expr),
    share_reads (rename_held r held) (rename_reads_made r made)
      (rename_expr r e)
    = rename_expr r (share_reads held made e).

Definition share_static_reads_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    share_static_reads (rename_expr r e)
    = rename_expr r (share_static_reads e).

(* A [Let] of a store whose first argument is a symbol has no value:
   no step of the machine writes a field of what a symbol is bound
   to. *)
Definition store_to_symbol_undefined_statement : Prop :=
  ∀ (statics : static_env) (hp : heap) (rd : read_record) (vars : env)
    (kind : block_access_kind) (init : init_or_assign) (field : Z)
    (s : symbol) (new : simple),
    exec_named statics hp rd vars
      (Prim (Heap (Block_set kind init field)) [Symbol s; new])
    = None.

(* Nor has the primitive one, on values the first of which is a
   symbol. *)
Definition exec_prim_store_to_symbol_statement : Prop :=
  ∀ (statics : static_env) (hp : heap) (kind : block_access_kind)
    (init : init_or_assign) (field : Z) (s : symbol) (new : value),
    exec_prim statics hp (Heap (Block_set kind init field))
      [Value_symbol s; new]
    = None.

(* ======================================================================== *)
(* Eight units                                                              *)
(* ======================================================================== *)

(* What the rewrite makes of eight small units, computed. Each binds
   [x] to 1 and [y] to 2, then one group: the code [reader], a closure
   of that code at the symbol [closure], and a block at the symbol
   [block]. The two reads are the projection of the value slot of the
   closure and the load of the first field of the block. *)
Section Examples.

Let x : variable := 1%positive.
Let y : variable := 2%positive.
Let arg : variable := 3%positive.
Let my_closure : variable := 4%positive.
Let first_read : variable := 5%positive.
Let second_read : variable := 6%positive.
Let stored : variable := 7%positive.
Let ret : continuation := 1%positive.
Let exn : continuation := 2%positive.
Let done : continuation := 3%positive.
Let after : continuation := 4%positive.
Let reader : code_id := 1%positive.
Let closure : symbol := 1%positive.
Let block : symbol := 2%positive.
Let slot : value_slot := 1%positive.

Definition example_arity : arity := [Singleton Value].

Definition example_slot_read : named :=
  Project_value_slot slot (Symbol closure).

Definition example_field_read : named :=
  Prim
    (Unary (Block_load (Block_access_values None None Any_value) Immutable 0))
    [Symbol block].

(* Two reads, and then the continuation given applied to what they
   bound. *)
Definition example_reads (second : named) (k : continuation) : expr :=
  Let first_read Value example_slot_read
    (Let second_read Value second
       (Apply_cont
          (Apply_cont_expr k [Var first_read; Var second_read] None))).

(* The code, which makes the two reads, with the second as given. *)
Definition example_code (second : named) : code :=
  Code [arg] my_closure ret exn
    (make_code_metadata example_arity [Singleton Value; Singleton Value])
    (example_reads second ret).

(* The unit: the closure has the value slots given, the block the
   field given, and the body is what follows the group. *)
Definition example_unit (c : code) (slots : list (value_slot * simple))
    (in_field : simple) (body : expr) : expr :=
  Let x Value (Simple (Const (Tagged_immediate 1)))
    (Let y Value (Simple (Const (Tagged_immediate 2)))
       (Let_static
          [Bind_code reader c;
           Bind_symbol closure (Static_const_set_of_closures reader slots);
           Bind_symbol block (Static_const_block 0 [in_field])]
          body)).

Definition example_exit : expr :=
  Apply_cont (Apply_cont_expr done [Symbol block] None).

(* Where the slot and the field hold one variable, the second read of
   the code is the variable of the first. *)
Definition share_static_reads_shares_statement : Prop :=
  share_static_reads
    (example_unit (example_code example_field_read) [(slot, Var x)] (Var x)
       example_exit)
  = example_unit (example_code (Simple (Var first_read))) [(slot, Var x)]
      (Var x) example_exit.

(* Where they hold two variables, nothing is shared. This is the unit
   of which a term after that shared the two reads would be another
   program. *)
Definition share_static_reads_two_variables_statement : Prop :=
  let u :=
    example_unit (example_code example_field_read) [(slot, Var x)] (Var y)
      example_exit
  in
  share_static_reads u = u.

(* A closure whose constant lists its slot twice, first with [x] and
   then with [y], holds [x] in the slot, and the field holds [y]:
   nothing is shared, and the two reads, made after the group, give 1
   and 2. *)
Definition share_static_reads_slot_twice_statement : Prop :=
  let u :=
    example_unit (example_code example_field_read)
      [(slot, Var x); (slot, Var y)] (Var y)
      (example_reads example_field_read done)
  in
  share_static_reads u = u
  ∧ run 6 (State u [] [] empty_static_env empty_heap no_reads)
    = Exit done
        [Value_const (Tagged_immediate 1); Value_const (Tagged_immediate 2)]
        empty_heap no_reads.

(* Where the closure and the block are bound by two groups, nothing is
   shared, though both hold [x]: the classes are of one group. *)
Definition share_static_reads_two_groups_statement : Prop :=
  let u :=
    Let x Value (Simple (Const (Tagged_immediate 1)))
      (Let_static
         [Bind_code reader (example_code example_field_read);
          Bind_symbol closure
            (Static_const_set_of_closures reader [(slot, Var x)])]
         (Let_static [Bind_symbol block (Static_const_block 0 [Var x])]
            example_exit))
  in
  share_static_reads u = u.

(* Where a handler binds the block again, with another variable, the
   check fails and the unit is left as it is. *)
Definition share_static_reads_bound_again_statement : Prop :=
  let u :=
    Let_cont_non_recursive after
      (Handler [] (make_handler_metadata [])
         (Let_static [Bind_symbol block (Static_const_block 0 [Var y])]
            example_exit))
      (example_unit (example_code example_field_read) [(slot, Var x)] (Var x)
         example_exit)
  in
  groups_hold (places_held u) u = false ∧ share_static_reads u = u.

(* The same code, marked as a stub. *)
Definition example_stub (c : code) : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    Code params my_closure ret exn (stub_code_metadata metadata) body
  end.

(* The code of a stub is left as it is. *)
Definition share_static_reads_in_stub_statement : Prop :=
  let u :=
    example_unit (example_stub (example_code example_field_read))
      [(slot, Var x)] (Var x) example_exit
  in
  share_static_reads u = u.

(* A read made before a handler is defined is known in the handler,
   unless a parameter of the handler has its variable. *)
Definition example_handler (param : variable) (second : named) : expr :=
  example_unit (example_code (Simple (Var first_read))) [(slot, Var x)]
    (Var x)
    (Let first_read Value example_slot_read
       (Let_cont_non_recursive after
          (Handler [param] (make_handler_metadata [Value])
             (Let second_read Value second
                (Apply_cont
                   (Apply_cont_expr done [Var first_read; Var second_read]
                      None))))
          (Apply_cont (Apply_cont_expr after [Var y] None)))).

Definition share_static_reads_in_handler_statement : Prop :=
  share_static_reads (example_handler arg example_field_read)
  = example_handler arg (Simple (Var first_read))
  ∧ share_static_reads (example_handler first_read example_field_read)
    = example_handler first_read example_field_read.

(* A unit that writes the field of the block, between the group and
   the two reads. The check does not look at the store, and the reads
   are shared; the unit is undefined where it writes, at its fourth
   step, so the refinement says nothing of it. *)
Definition share_static_reads_store_statement : Prop :=
  let store :=
    Prim (Heap (Block_set (Block_access_values None None Any_value)
                  Assignment 0))
      [Symbol block; Var y]
  in
  let u (second : named) :=
    example_unit (example_code (Simple (Var first_read))) [(slot, Var x)]
      (Var x)
      (Let stored Value store (example_reads second done))
  in
  share_static_reads (u example_field_read) = u (Simple (Var first_read))
  ∧ run 4
    (State (u example_field_read) [] [] empty_static_env empty_heap no_reads)
    = Undefined.

End Examples.

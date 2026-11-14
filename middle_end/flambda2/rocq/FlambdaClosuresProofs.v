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


(* Proofs of the statements of FlambdaClosuresSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Arith Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaBindersSpec FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaApplySpec FlambdaApplyProofs
  FlambdaClosuresSpec.
Import ListNotations.

(* ======================================================================== *)
(* Lists                                                                    *)
(* ======================================================================== *)

(* In a list of bindings of distinct names, a binding is the one found. *)
Lemma lookup_NoDup_in {A : Type} (bindings : list (positive * A))
    (name : positive) (a : A) :
  NoDup (map fst bindings) -> (name, a) ∈ bindings ->
  lookup name bindings = Some a.
Proof.
  induction bindings as [ | [bound b] bindings IH]; simpl; [ intros _ [] | ].
  intros Hdistinct [Heq | Hin].
  - injection Heq as -> ->. rewrite Pos.eqb_refl. reflexivity.
  - inversion Hdistinct as [ | ? ? Hfresh Hrest]. subst.
    destruct (Pos.eqb_spec name bound) as [-> | _]; [ | exact (IH Hrest Hin) ].
    exfalso. apply Hfresh. change bound with (fst (bound, a)).
    apply in_map, Hin.
Qed.

(* A name bound in a map has something bound to it. *)
Lemma lookup_bound {A : Type} (name : positive)
    (bindings : list (positive * A)) :
  name ∈ map fst bindings -> ∃ a : A, lookup name bindings = Some a.
Proof.
  intros Hin. destruct (lookup name bindings) as [a | ] eqn:Hlookup;
    [ eauto | ].
  destruct (lookup_none_not_in name bindings Hlookup Hin).
Qed.

Lemma combine_fst_length {A B C : Type} (l : list A) (l' : list (B * C)) :
  length l = length l' -> length (combine l (map fst l')) = length l.
Proof.
  revert l'. induction l as [ | a l IH]; intros [ | b l']; simpl;
    try discriminate; [ reflexivity | ].
  intros [= Hlen]. rewrite (IH l' Hlen). reflexivity.
Qed.

(* ======================================================================== *)
(* A set of closures made by a [Let]                                        *)
(* ======================================================================== *)

Section Let_closures.

Variable decls : list (function_slot * code_id).

Variable slot_values : list (value_slot * value).

(* What [let_projections] binds, the first bound first. *)
Definition projections_bound (bound : list (variable * function_slot)) : env :=
  map (fun '(y, slot) => (y, Value_closure_of_set slot decls slot_values))
    bound.

(* The projections are bound one at a time, each from the closure [base],
   which stays in scope as none of the variables bound is [base]. *)
Lemma let_projections_runs (base : variable) (base_slot : function_slot)
    (code : code_id) (body : expr) (conts : cont_env) (statics : static_env)
      (hp : heap) (rd : read_record) :
  lookup base_slot decls = Some code ->
  ∀ (bound : list (variable * function_slot)) (vars : env),
    lookup base vars
    = Some (Value_closure_of_set base_slot decls slot_values) ->
    (∀ (y : variable) (slot : function_slot),
      (y, slot) ∈ bound ->
      y <> base ∧ ∃ code' : code_id, lookup slot decls = Some code') ->
    run (length bound)
      (State (let_projections base base_slot bound body) vars conts statics hp
         rd)
    = Step
        (State body (rev (projections_bound bound) ++ vars) conts statics hp
           rd).
Proof.
  intros Hfrom.
  induction bound as [ | [y slot] bound IH]; intros vars Hbase Hbound;
    [ reflexivity | ].
  destruct (Hbound y slot (or_introl eq_refl)) as [Hy [code' Hto]].
  simpl length. rewrite run_succ. simpl let_projections.
  rewrite step_let_project_function_slot. simpl eval_simple. rewrite Hbase.
  rewrite (project_function_slot_of_set statics base_slot slot decls
             slot_values code code' Hfrom Hto).
  cbn [kind_of_value]. rewrite kind_eqb_refl, IH.
  - simpl. rewrite <- app_assoc. reflexivity.
  - rewrite lookup_other; [ exact Hbase | congruence ].
  - intros y' slot' Hin. apply Hbound. right. exact Hin.
Qed.

(* The bindings of [closures_bound] for the variables [xs] and the
   functions [ds] of the set, the first bound first. *)
Definition closures_bound_to (xs : list variable)
    (ds : list (function_slot * code_id)) : env :=
  map (fun '(x, (slot, _)) => (x, Value_closure_of_set slot decls slot_values))
    (combine xs ds).

Lemma projections_bound_combine (xs : list variable)
    (ds : list (function_slot * code_id)) :
  projections_bound (combine xs (map fst ds)) = closures_bound_to xs ds.
Proof.
  unfold projections_bound, closures_bound_to. revert ds.
  induction xs as [ | x xs IH]; intros [ | [slot code] ds]; simpl;
    try reflexivity.
  rewrite IH. reflexivity.
Qed.

Lemma closures_bound_to_names (xs : list variable)
    (ds : list (function_slot * code_id)) (x : variable) :
  x ∈ map fst (closures_bound_to xs ds) -> x ∈ xs.
Proof.
  revert ds. induction xs as [ | x' xs IH]; intros [ | [slot code] ds];
    simpl; try contradiction.
  intros [Heq | Hin]; [ left; exact Heq | right; exact (IH ds Hin) ].
Qed.

Lemma closures_bound_to_distinct (xs : list variable)
    (ds : list (function_slot * code_id)) :
  NoDup xs -> NoDup (map fst (closures_bound_to xs ds)).
Proof.
  revert ds. induction xs as [ | x xs IH]; intros [ | [slot code] ds];
    simpl; try (intros _; apply NoDup_nil).
  intros Hdistinct. inversion Hdistinct as [ | ? ? Hfresh Hrest]. subst.
  apply NoDup_cons; [ | exact (IH ds Hrest) ].
  intros Hin. exact (Hfresh (closures_bound_to_names xs ds x Hin)).
Qed.

Lemma closures_bound_to_nth (xs : list variable)
    (ds : list (function_slot * code_id)) :
  ∀ (i : nat) (x : variable) (slot : function_slot) (code : code_id),
    nth_error xs i = Some x ->
    nth_error ds i = Some (slot, code) ->
    (x, Value_closure_of_set slot decls slot_values)
    ∈ closures_bound_to xs ds.
Proof.
  revert ds. induction xs as [ | x' xs IH]; intros [ | [slot' code'] ds];
    intros [ | i] x slot code; simpl; try discriminate.
  - intros [= <-] [= <- <-]. left. reflexivity.
  - intros Hx Hslot. right. exact (IH ds i x slot code Hx Hslot).
Qed.

(* A variable of a [Let] of a set is bound, in its scope, to the closure
   at the slot of the function in its position. *)
Lemma closures_bound_to_lookup (xs : list variable)
    (ds : list (function_slot * code_id)) (vars : env) (i : nat)
    (x : variable) (slot : function_slot) (code : code_id) :
  NoDup xs ->
  nth_error xs i = Some x ->
  nth_error ds i = Some (slot, code) ->
  lookup x (rev (closures_bound_to xs ds) ++ vars)
  = Some (Value_closure_of_set slot decls slot_values).
Proof.
  intros Hdistinct Hx Hslot. rewrite lookup_app.
  rewrite (lookup_NoDup_in (rev (closures_bound_to xs ds)) x
             (Value_closure_of_set slot decls slot_values));
    [ reflexivity | | ].
  - rewrite map_rev. apply NoDup_rev, closures_bound_to_distinct, Hdistinct.
  - apply -> in_rev. exact (closures_bound_to_nth xs ds i x slot code Hx Hslot).
Qed.

End Let_closures.

(* The function at a position of a set has code. *)
Lemma nth_error_lookup (decls : list (function_slot * code_id)) (i : nat)
    (slot : function_slot) (code : code_id) :
  nth_error decls i = Some (slot, code) ->
  ∃ code' : code_id, lookup slot decls = Some code'.
Proof.
  intros Hnth. apply lookup_bound. change slot with (fst (slot, code)).
  apply in_map. exact (nth_error_In decls i Hnth).
Qed.

Theorem let_closures_runs : let_closures_runs_statement.
Proof.
  intros bound_vars decls value_slots body e vars conts statics hp rd
    slot_values
    Hlet Hdistinct Hslots.
  unfold let_closures in Hlet.
  destruct bound_vars as [ | x xs]; [ discriminate | ].
  destruct decls as [ | [slot code] others]; [ discriminate | ].
  destruct (Nat.eqb (length xs) (length others)) eqn:Hlen; [ | discriminate ].
  injection Hlet as <-. apply Nat.eqb_eq in Hlen.
  apply distinct_NoDup in Hdistinct.
  inversion Hdistinct as [ | ? ? Hfresh Hrest]. subst.
  simpl length. rewrite run_succ, step_let_closure_of_set, lookup_head, Hslots.
  rewrite kind_eqb_refl.
  rewrite <- (combine_fst_length xs others Hlen) at 1.
  rewrite (let_projections_runs ((slot, code) :: others) slot_values x slot
             code body conts statics hp rd_ (lookup_head _ _ _)
             (combine xs (map fst others)));
    [ | apply lookup_head | ].
  - rewrite projections_bound_combine. unfold closures_bound. simpl.
    rewrite <- app_assoc. reflexivity.
  - intros y slot' Hin. split.
    + intros ->. apply Hfresh. exact (in_combine_l _ _ _ _ Hin).
    + apply lookup_bound. simpl. right. exact (in_combine_r _ _ _ _ Hin).
Qed.

Theorem let_closures_undefined : let_closures_undefined_statement.
Proof.
  intros bound_vars decls value_slots body e vars conts statics hp rd Hlet
    Hslots.
  unfold let_closures in Hlet.
  destruct bound_vars as [ | x xs]; [ discriminate | ].
  destruct decls as [ | [slot code] others]; [ discriminate | ].
  destruct (Nat.eqb (length xs) (length others)); [ | discriminate ].
  injection Hlet as <-.
  rewrite step_let_closure_of_set, lookup_head, Hslots. reflexivity.
Qed.

(* [closures_bound] is the bindings of [closures_bound_to], the last bound
   first. *)
Lemma closures_bound_rev (bound_vars : list variable)
    (decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)) :
  closures_bound bound_vars decls slot_values
  = rev (closures_bound_to decls slot_values bound_vars decls).
Proof. reflexivity. Qed.

(* Both sides bind the closures of the set, or neither does; then the
   projection is the closure that the variable of its slot is bound to. *)
Theorem project_function_slot_let_closures :
  project_function_slot_let_closures_statement.
Proof.
  intros bound_vars decls value_slots move_from move_to x_from x_to y kind
    code code' i j body e e' Hdistinct Hx_from Hfrom Hx_to Hto Hlet Hlet'
    vars conts statics hp rd.
  destruct (eval_value_slots vars value_slots) as [slot_values | ] eqn:Hslots.
  - pose proof (let_closures_runs bound_vars decls value_slots _ e vars conts
                  statics hp rd slot_values Hlet Hdistinct Hslots) as Hrun.
    pose proof (let_closures_runs bound_vars decls value_slots _ e' vars conts
                  statics hp rd slot_values Hlet' Hdistinct Hslots) as Hrun'.
    eapply equivalent_trans;
      [ exact (equivalent_of_run (length bound_vars) 0 _ _ Hrun) | ].
    eapply equivalent_trans;
      [ | apply equivalent_sym;
          exact (equivalent_of_run (length bound_vars) 0 _ _ Hrun') ].
    apply let_equivalent_step_not_prim;
      [ intros ? ?; discriminate | intros ? ?; discriminate | | intros v _ _;
       apply equivalent_refl ].
    apply distinct_NoDup in Hdistinct. rewrite closures_bound_rev. simpl.
    rewrite (closures_bound_to_lookup decls slot_values bound_vars decls vars
               i x_from move_from code Hdistinct Hx_from Hfrom).
    rewrite (closures_bound_to_lookup decls slot_values bound_vars decls vars
               j x_to move_to code' Hdistinct Hx_to Hto).
    destruct (nth_error_lookup decls i move_from code Hfrom) as [c Hc].
    destruct (nth_error_lookup decls j move_to code' Hto) as [c' Hc'].
    symmetry.
    exact (project_function_slot_of_set statics move_from move_to decls
             slot_values c c' Hc Hc').
  - apply step_eq_equivalent.
    rewrite (let_closures_undefined bound_vars decls value_slots _ e vars
               conts statics _ rd_ Hlet Hslots).
    rewrite (let_closures_undefined bound_vars decls value_slots _ e' vars
               conts statics _ rd_ Hlet' Hslots).
    reflexivity.
Qed.

(* The variable is bound to the closure at the slot projected, whose code
   is that of the slot. *)
Theorem apply_sibling : apply_sibling_statement.
Proof.
  intros x kind move_from move_to closure args k exn extra args_arity
    return_arity decls slot_values code vars conts statics hp rd Hclosure Hto.
  apply let_equivalent_step_not_prim;
    [ intros ? ?; discriminate | intros ? ?; discriminate | reflexivity | ].
  intros v. simpl. rewrite Hclosure. unfold project_function_slot.
  rewrite Pos.eqb_refl, Hto.
  destruct (lookup move_from decls); [ | discriminate ].
  simpl. intros [= <-] _.
  apply (apply_indirect_known (Var x) args extra k exn args_arity
           return_arity (Value_closure_of_set move_to decls slot_values) code
           slot_values).
  - simpl. rewrite Pos.eqb_refl. reflexivity.
  - simpl. rewrite Hto. reflexivity.
Qed.

(* ======================================================================== *)
(* A static set of closures                                                 *)
(* ======================================================================== *)

(* Each binding of [bind_closures] binds its symbol, which was not bound,
   and the bindings that follow leave it so. [all] is all the symbols of
   the set, and [symbols] those still to bind. *)
Lemma bind_closures_binds_aux (vars : env)
    (all : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple))
    (slot_values : list (value_slot * value)) :
  eval_value_slots vars value_slots = Some slot_values ->
  ∀ (symbols : list (function_slot * symbol)) (statics extended : static_env),
    eval_static_bindings vars
      (map
         (fun '(slot, s) =>
            Bind_symbol s
              (Static_const_closure_of_set slot all decls value_slots))
         symbols)
      statics
    = Some extended ->
    ∀ (slot : function_slot) (s : symbol),
      (slot, s) ∈ symbols ->
      lookup s (static_values extended)
      = Some (Static_closure_of_set slot all decls slot_values).
Proof.
  intros Hslots.
  induction symbols as [ | [slot0 s0] symbols IH]; intros statics extended;
    simpl; [ intros _ slot s [] | ].
  destruct (lookup s0 (static_values statics)); [ discriminate | ].
  destruct (lookup slot0 decls) as [code0 | ]; [ | discriminate ].
  simpl. rewrite Hslots. simpl. intros Heval slot s [Heq | Hin].
  - injection Heq as <- <-.
    apply (proj2 (eval_static_bindings_extend vars _ _ _ Heval)). simpl.
    apply lookup_head.
  - exact (IH _ _ Heval slot s Hin).
Qed.

Theorem bind_closures_binds : bind_closures_binds_statement.
Proof.
  intros vars closure_symbols decls value_slots slot_values statics extended
    Hslots Heval.
  exact (bind_closures_binds_aux vars closure_symbols decls value_slots
           slot_values Hslots closure_symbols statics extended Heval).
Qed.

Theorem static_set_bound_extend : static_set_bound_extend_statement.
Proof.
  intros statics statics' closure_symbols decls slot_values Hbound
    [_ Hvalues] slot s Hin.
  exact (Hvalues s _ (Hbound slot s Hin)).
Qed.

Theorem project_static_set : project_static_set_statement.
Proof.
  intros statics closure_symbols decls slot_values move_from move_to s s' code
    code' Hbound Hin Hsymbol Hfrom Hto.
  pose proof (Hbound move_from s Hin) as Hs.
  pose proof (Hbound move_to s' (lookup_in _ _ _ Hsymbol)) as Hs'.
  split.
  - unfold project_function_slot.
    rewrite Hs, Pos.eqb_refl, Hfrom, Hto, Hsymbol. reflexivity.
  - unfold closure_in_set. rewrite Hs', Hto. reflexivity.
Qed.

(* A set with a symbol is not bound where a value slot has no value. *)
Lemma bind_closures_undefined (vars : env)
    (closure_symbols : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (statics : static_env)
    (slot : function_slot) (s : symbol) :
  (slot, s) ∈ closure_symbols ->
  eval_value_slots vars value_slots = None ->
  eval_static_bindings vars (bind_closures closure_symbols decls value_slots)
    statics
  = None.
Proof.
  intros Hin Hslots. unfold bind_closures.
  destruct closure_symbols as [ | [slot0 s0] symbols]; [ destruct Hin | ].
  simpl. destruct (lookup s0 (static_values statics)); [ reflexivity | ].
  destruct (lookup slot0 decls); [ | reflexivity ].
  simpl. rewrite Hslots. reflexivity.
Qed.

(* Both sides bind the set, or neither does; once it is bound the
   projection is the symbol. *)
Theorem project_function_slot_static :
  project_function_slot_static_statement.
Proof.
  intros closure_symbols decls value_slots move_from move_to s s' code code' x
    kind body Hin Hsymbol Hfrom Hto vars conts statics hp rd.
  apply let_static_equivalent_step.
  destruct
    (eval_static_bindings vars
       (bind_closures closure_symbols decls value_slots) statics)
    as [extended | ] eqn:Heval; [ | exact I ].
  destruct (eval_value_slots vars value_slots) as [slot_values | ] eqn:Hslots;
    [ | rewrite (bind_closures_undefined vars closure_symbols decls
                   value_slots statics move_from s Hin Hslots) in Heval;
        discriminate ].
  pose proof (bind_closures_binds vars closure_symbols decls value_slots
                slot_values statics extended Hslots Heval) as Hbound.
  destruct (project_static_set extended closure_symbols decls slot_values
              move_from move_to s s' code code' Hbound Hin Hsymbol Hfrom Hto)
    as [Hproject _].
  apply let_equivalent_step_not_prim;
    [ intros ? ?; discriminate | intros ? ?; discriminate | | intros v _ _;
     apply equivalent_refl ].
  cbn [eval_named eval_simple]. rewrite Hproject. reflexivity.
Qed.

(* The value of a value slot of a set is the value, where the set was
   made, of what the set gives the slot. *)
Lemma eval_value_slots_lookup (vars : env) (slot : value_slot)
    (contents : simple) :
  ∀ (value_slots : list (value_slot * simple))
    (slot_values : list (value_slot * value)),
    eval_value_slots vars value_slots = Some slot_values ->
    lookup slot value_slots = Some contents ->
    lookup slot slot_values = eval_simple vars contents.
Proof.
  induction value_slots as [ | [slot' s] value_slots IH]; intros slot_values;
    simpl; [ discriminate | ].
  destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
  destruct (eval_value_slots vars value_slots) as [rest | ];
    [ | discriminate ].
  simpl. intros [= <-]. simpl. destruct (Pos.eqb slot slot').
  - intros [= <-]. symmetry. exact Hs.
  - exact (IH rest eq_refl).
Qed.

(* Both sides bind the set, or neither does; once it is bound the slot
   holds the value that its contents had, and still have, the variables
   being the same. *)
Theorem project_value_slot_static : project_value_slot_static_statement.
Proof.
  intros closure_symbols decls value_slots project_from s code slot contents
    x kind body Hin Hfrom Hslot vars conts statics hp rd.
  apply let_static_equivalent_step.
  destruct
    (eval_static_bindings vars
       (bind_closures closure_symbols decls value_slots) statics)
    as [extended | ] eqn:Heval; [ | exact I ].
  destruct (eval_value_slots vars value_slots) as [slot_values | ] eqn:Hslots;
    [ | rewrite (bind_closures_undefined vars closure_symbols decls
                   value_slots statics project_from s Hin Hslots) in Heval;
        discriminate ].
  pose proof (bind_closures_binds vars closure_symbols decls value_slots
                slot_values statics extended Hslots Heval project_from s Hin)
    as Hs.
  apply let_equivalent_step_not_prim;
    [ intros ? ?; discriminate | intros ? ?; discriminate | | intros v _ _;
     apply equivalent_refl ].
  cbn [eval_named eval_simple]. unfold project_value_slot, closure_in_set.
  rewrite Hs, Hfrom. cbn. rewrite Pos.eqb_refl. symmetry.
  exact (eval_value_slots_lookup vars slot contents value_slots slot_values
           Hslots Hslot).
Qed.

(* Two bindings to one thing, where each thing is bound once, are of one
   name. *)
Lemma in_snd_NoDup {A : Type} (bindings : list (positive * A))
    (name name' : positive) (a : A) :
  NoDup (map snd bindings) ->
  (name, a) ∈ bindings -> (name', a) ∈ bindings -> name = name'.
Proof.
  induction bindings as [ | [bound b] bindings IH]; simpl; [ intros _ [] | ].
  intros Hdistinct. inversion Hdistinct as [ | ? ? Hfresh Hrest]. subst.
  intros [Heq | Hin] [Heq' | Hin'].
  - congruence.
  - injection Heq as -> ->. exfalso. apply Hfresh.
    change a with (snd (name', a)). apply in_map, Hin'.
  - injection Heq' as -> ->. exfalso. apply Hfresh.
    change a with (snd (name, a)). apply in_map, Hin.
  - exact (IH Hrest Hin Hin').
Qed.

Theorem closure_code_determines_slot :
  closure_code_determines_slot_statement.
Proof.
  intros statics v slot slot' decls slots slots' code Hin_set Hdistinct Hof
    Hslot'.
  apply (proj1 (closure_in_set_of statics v code slots')) in Hof
    as (slot0 & decls0 & Hin_set0 & Hslot0).
  rewrite Hin_set in Hin_set0. injection Hin_set0 as <- <- _.
  apply distinct_NoDup in Hdistinct.
  exact (in_snd_NoDup decls slot' slot code Hdistinct
           (lookup_in _ _ _ Hslot') (lookup_in _ _ _ Hslot0)).
Qed.

(* ======================================================================== *)
(* The sets of a well-formed unit                                           *)
(* ======================================================================== *)

(* A symbol bound after a group was bound before it, or the group binds
   it. *)
Lemma eval_static_bindings_symbol (vars : env)
    (bindings : list static_binding) :
  ∀ (statics extended : static_env) (s : symbol)
    (allocated : statically_allocated_value),
    eval_static_bindings vars bindings statics = Some extended ->
    lookup s (static_values extended) = Some allocated ->
    lookup s (static_values statics) = Some allocated
    ∨ ∃ const : static_const,
        Bind_symbol s const ∈ bindings
        ∧ eval_static_const vars const = Some allocated.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics extended s allocated; simpl.
  - intros [= <-]. auto.
  - destruct (lookup id (static_code statics)); [ discriminate | ].
    intros Heval Hs.
    destruct (IH _ _ s allocated Heval Hs) as [Hold | (const & Hin & Hconst)];
      [ left; exact Hold | right; exists const; auto ].
  - intros Heval Hs.
    destruct (IH _ _ s allocated Heval Hs) as [Hold | (const & Hin & Hconst)];
      [ left; exact Hold | right; exists const; auto ].
  - destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [a | ] eqn:Hconst; simpl;
      [ | discriminate ].
    intros Heval Hs.
    destruct (IH _ _ s allocated Heval Hs)
      as [Hold | (const' & Hin & Hconst')];
      [ | right; exists const'; auto ].
    simpl in Hold.
    destruct (Pos.eqb_spec s sym) as [-> | _]; [ | left; exact Hold ].
    injection Hold as <-. right. exists const. auto.
Qed.

(* A group binds each of its symbols to what its constant is. *)
Lemma eval_static_bindings_binds (vars : env)
    (bindings : list static_binding) :
  ∀ (statics extended : static_env) (s : symbol) (const : static_const),
    eval_static_bindings vars bindings statics = Some extended ->
    Bind_symbol s const ∈ bindings ->
    ∃ allocated : statically_allocated_value,
      eval_static_const vars const = Some allocated
      ∧ lookup s (static_values extended) = Some allocated.
Proof.
  induction bindings as [ | [id c | id | sym const0] bindings IH];
    intros statics extended s const; simpl; [ intros _ [] | | | ].
  - destruct (lookup id (static_code statics)); [ discriminate | ].
    intros Heval [[=] | Hin]. exact (IH _ _ s const Heval Hin).
  - intros Heval [[=] | Hin]. exact (IH _ _ s const Heval Hin).
  - destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const0) as [a | ] eqn:Hconst; simpl;
      [ | discriminate ].
    intros Heval [Heq | Hin]; [ | exact (IH _ _ s const Heval Hin) ].
    injection Heq as <- <-. exists a. split; [ exact Hconst | ].
    apply (proj2 (eval_static_bindings_extend vars _ _ _ Heval)). simpl.
    apply lookup_head.
Qed.

(* Only the constant of a closure of a static set makes one. *)
Lemma eval_static_const_closure_of_set (vars : env) (const : static_const)
    (slot : function_slot) (symbols : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)) :
  eval_static_const vars const
  = Some (Static_closure_of_set slot symbols decls slot_values) ->
  ∃ value_slots : list (value_slot * simple),
    const = Static_const_closure_of_set slot symbols decls value_slots
    ∧ eval_value_slots vars value_slots = Some slot_values.
Proof.
  destruct const
    as [code slots | fslot syms fdecls slots | tag fields | contents | bytes
        | elements];
    simpl.
  - destruct (eval_value_slots vars slots); discriminate.
  - destruct (lookup fslot fdecls); [ | discriminate ]. simpl.
    destruct (eval_value_slots vars slots) as [evaluated | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <- <- <- <-]. exists slots. auto.
  - destruct (eval_simples vars fields); discriminate.
  - destruct (eval_simple vars contents); discriminate.
  - discriminate.
  - destruct (eval_simples vars elements); discriminate.
Qed.

Lemma binds_symbol_to_sound (s : symbol) (const : static_const)
    (binding : static_binding) :
  binds_symbol_to s const binding = true -> binding = Bind_symbol s const.
Proof.
  destruct binding as [id c | id | s' const']; simpl; try discriminate.
  intros H. apply andb_true_iff in H as [Hs Hconst].
  apply Pos.eqb_eq in Hs.
  destruct (static_const_eq_dec const const'); [ | discriminate ].
  congruence.
Qed.

(* What [static_set_whole] asks of the binding of a closure of a set. *)
Lemma static_set_whole_facts (bindings : list static_binding) (s : symbol)
    (slot : function_slot) (symbols : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) :
  static_set_whole bindings
    (Bind_symbol s
       (Static_const_closure_of_set slot symbols decls value_slots))
  = true ->
  is_bound slot decls = true
  ∧ lookup slot symbols = Some s
  ∧ ∀ (slot' : function_slot) (s' : symbol),
      (slot', s') ∈ symbols ->
      Bind_symbol s'
        (Static_const_closure_of_set slot' symbols decls value_slots)
      ∈ bindings.
Proof.
  unfold static_set_whole. intros H.
  apply andb_true_iff in H as [H Hall].
  apply andb_true_iff in H as [H Hsymbol].
  apply andb_true_iff in H as [Hset _]. unfold set_well_formed in Hset.
  apply andb_true_iff in Hset as [Hset _].
  apply andb_true_iff in Hset as [Hslot _].
  split; [ exact Hslot | ]. split.
  - destruct (lookup slot symbols) as [s0 | ]; [ | discriminate ].
    apply Pos.eqb_eq in Hsymbol. congruence.
  - intros slot' s' Hin. rewrite forallb_forall in Hall.
    specialize (Hall (slot', s') Hin). simpl in Hall.
    apply existsb_exists in Hall as [binding [Hbinding Hbinds]].
    apply binds_symbol_to_sound in Hbinds. subst binding. exact Hbinding.
Qed.

(* A group every set of which is whole, added to a static environment in
   which the sets are bound: the sets are bound. *)
Lemma eval_static_bindings_sets_bound (vars : env)
    (bindings : list static_binding) (statics extended : static_env) :
  forallb (static_set_whole bindings) bindings = true ->
  static_sets_bound statics ->
  eval_static_bindings vars bindings statics = Some extended ->
  static_sets_bound extended.
Proof.
  intros Hwhole Hbound Heval s slot symbols decls slot_values Hs.
  pose proof (eval_static_bindings_extend vars bindings statics extended
                Heval) as Hextend.
  destruct (eval_static_bindings_symbol vars bindings statics extended s _
              Heval Hs) as [Hold | (const & Hin & Hconst)].
  - destruct (Hbound s slot symbols decls slot_values Hold)
      as [Hsymbol Hset].
    split; [ exact Hsymbol | ].
    exact (static_set_bound_extend statics extended symbols decls slot_values
             Hset Hextend).
  - destruct (eval_static_const_closure_of_set vars const slot symbols decls
                slot_values Hconst) as (value_slots & -> & Hslots).
    rewrite forallb_forall in Hwhole.
    destruct (static_set_whole_facts bindings s slot symbols decls
                value_slots (Hwhole _ Hin)) as (_ & Hsymbol & Hall).
    split; [ exact Hsymbol | ]. intros slot' s' Hin'.
    destruct (eval_static_bindings_binds vars bindings statics extended s' _
                Heval (Hall slot' s' Hin')) as (allocated & Hallocated & Hs').
    rewrite Hs'. f_equal. simpl in Hallocated. rewrite Hslots in Hallocated.
    destruct (lookup slot' decls); [ | discriminate ].
    simpl in Hallocated. congruence.
Qed.

(* [cont_sets_ok], [conts_sets_ok], [code_sets_ok] and [sets_inv] are in
   FlambdaClosuresSpec.v. *)

Lemma conts_sets_ok_app (conts conts' : cont_env) :
  conts_sets_ok conts -> conts_sets_ok conts' ->
  conts_sets_ok (conts ++ conts').
Proof.
  unfold conts_sets_ok. induction conts as [ | kc conts IH]; simpl; [ auto | ].
  intros [Hc Hrest] H'. split; [ exact Hc | exact (IH Hrest H') ].
Qed.

Lemma resolve_sets_ok (k : continuation) (conts : cont_env) :
  conts_sets_ok conts -> cont_sets_ok (resolve k conts).
Proof.
  unfold resolve, conts_sets_ok.
  induction conts as [ | [k' c] conts IH]; simpl; [ intros _; exact I | ].
  intros [Hc Hrest]. destruct (Pos.eqb k k'); [ exact Hc | exact (IH Hrest) ].
Qed.

Lemma rec_closures_sets_ok (vars : env) (conts : cont_env)
    (all handlers : list (continuation * continuation_handler)) :
  forallb (fun '(_, h) => handler_sets_well_formed h) all = true ->
  conts_sets_ok conts ->
  forallb (fun '(_, h) => handler_sets_well_formed h) handlers = true ->
  conts_sets_ok (map (fun '(k, h) => (k, Closure h vars conts all)) handlers).
Proof.
  intros Hall Hconts. unfold conts_sets_ok.
  induction handlers as [ | [k h] handlers IH]; simpl;
    [ intros _; exact I | ].
  intros H. apply andb_true_iff in H as [Hh Hrest].
  split; [ | exact (IH Hrest) ].
  split; [ exact Hh | ]. split; [ exact Hall | exact Hconts ].
Qed.

Lemma eval_static_bindings_code_sets_ok (vars : env)
    (bindings : list static_binding) :
  ∀ statics statics' : static_env,
    forallb binding_sets_well_formed bindings = true ->
    code_sets_ok statics ->
    eval_static_bindings vars bindings statics = Some statics' ->
    code_sets_ok statics'.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics statics'; simpl.
  - intros _ Hok [= <-]. exact Hok.
  - intros H Hok. apply andb_true_iff in H as [Hc Hrest].
    destruct (lookup id (static_code statics)); [ discriminate | ].
    apply (IH _ _ Hrest). intros id' c'. simpl.
    destruct (Pos.eqb id' id); [ intros [= <-]; exact Hc | apply Hok ].
  - intros H Hok. exact (IH _ _ H Hok).
  - intros H Hok.
    destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const); simpl; [ | discriminate ].
    apply (IH _ _ H). exact Hok.
Qed.

Lemma with_extra_args_sets_ok (c : cont_value) (extra : list value) :
  cont_sets_ok c -> cont_sets_ok (with_extra_args c extra).
Proof. destruct extra; intros Hc; exact Hc. Qed.

(* Where a call and the application of a continuation land
   (FlambdaSemanticsProofs.call_function_lands). *)
Lemma sets_inv_over (remaining : list value)
    (remaining_arity return_arity : arity) (return_to exn_to : cont_value) :
  cont_sets_ok
    (Over_application remaining remaining_arity return_arity return_to
       exn_to)
  <-> cont_sets_ok return_to ∧ cont_sets_ok exn_to.
Proof. reflexivity. Qed.

Lemma sets_inv_code (statics : static_env) (hp : heap) (rd : read_record) :
  code_sets_ok statics ->
  static_sets_bound statics ->
  ∀ (called : code_id) (c : code) (callee_vars : env)
    (return_to exn_to : cont_value),
    lookup called (static_code statics) = Some c ->
    cont_sets_ok return_to ->
    cont_sets_ok exn_to ->
    sets_inv
      (State (code_body c) callee_vars
         [(code_return_continuation c, return_to);
          (code_exn_continuation c, exn_to)]
         statics hp rd).
Proof.
  intros S B called c callee_vars return_to exn_to Hlookup Hreturn Hexn.
  unfold sets_inv. simpl_state.
  pose proof (S called c Hlookup) as Hc.
  destruct c as [params my_closure ret exn metadata body].
  simpl in Hc |- *.
  exact (conj Hc (conj (conj Hreturn (conj Hexn I)) (conj S B))).
Qed.

Lemma sets_inv_handler (statics : static_env) (hp : heap) (rd : read_record) :
  code_sets_ok statics ->
  static_sets_bound statics ->
  ∀ (params : list variable) (metadata : handler_metadata) (handler : expr)
    (def_vars : env) (def_conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (handler_vars : env),
    cont_sets_ok
      (Closure (Handler params metadata handler) def_vars def_conts
         recursive) ->
    sets_inv
      (State handler handler_vars
         (rec_closures recursive def_vars def_conts ++ def_conts) statics hp
           rd).
Proof.
  intros S B params metadata handler def_vars def_conts recursive
    handler_vars Cr.
  unfold sets_inv. simpl_state. simpl in Cr.
  destruct Cr as (Ph & Hrec & Cdef).
  refine (conj Ph (conj (conts_sets_ok_app _ _ _ Cdef) (conj S B))).
  exact (rec_closures_sets_ok def_vars def_conts recursive recursive Hrec
           Cdef Hrec).
Qed.

Theorem sets_inv_step : sets_inv_step_statement.
Proof.
  intros s s' (P & C & S & B) Hstep. rewrite (state_eta s) in Hstep.
  set (vars := state_vars s) in *. set (conts := state_conts s) in *.
  set (statics := state_statics s) in *.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step in Hstep.
  - (* Let *)
    destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
    injection Hstep as <-. unfold sets_inv. simpl_state. simpl in P.
    apply andb_true_iff in P as [_ Pbody].
    exact (conj Pbody (conj C (conj S B))).
  - (* Let_static *)
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Heval; [ | discriminate ].
    injection Hstep as <-. unfold sets_inv. simpl_state. simpl in P.
    apply andb_true_iff in P as [P Pbody].
    apply andb_true_iff in P as [P Pbindings].
    apply andb_true_iff in P as [Pwhole _].
    refine (conj Pbody (conj C (conj _ _))).
    + exact (eval_static_bindings_code_sets_ok vars bindings statics statics'
               Pbindings S Heval).
    + exact (eval_static_bindings_sets_bound vars bindings statics statics'
               Pwhole B Heval).
  - (* Let_cont_non_recursive *)
    injection Hstep as <-. unfold sets_inv. simpl_state. simpl in P.
    apply andb_true_iff in P as [Ph Pbody].
    refine (conj Pbody (conj (conj _ C) (conj S B))).
    simpl. exact (conj Ph (conj eq_refl C)).
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers); [ discriminate | ].
    injection Hstep as <-. unfold sets_inv. simpl_state. simpl in P.
    apply andb_true_iff in P as [Phandlers Pbody].
    refine (conj Pbody (conj (conts_sets_ok_app _ _ _ C) (conj S B))).
    exact (rec_closures_sets_ok vars conts handlers handlers Phandlers C
             Phandlers).
  - (* Apply *)
    destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | discriminate ].
    refine
      (call_function_lands statics _ rd_ cont_sets_ok sets_inv
         (fun c extra H => H) sets_inv_over (sets_inv_code statics _ rd_ S B)
         (sets_inv_handler statics _ rd_ S B) _ _ _ _ _ _ _ s'
         (resolve_sets_ok _ conts C) _ Hstep).
    apply with_extra_args_sets_ok, resolve_sets_ok, C.
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    unfold apply_continuation in Hstep.
    exact
      (apply_cont_value_lands statics _ rd_ cont_sets_ok sets_inv
         (fun c extra H => H) sets_inv_over (sets_inv_code statics _ rd_ S B)
         (sets_inv_handler statics _ rd_ S B) _ vs s'
           (resolve_sets_ok _ conts C)
         Hstep).
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms); [ | discriminate ].
    injection Hstep as <-. unfold sets_inv. simpl_state.
    exact (conj eq_refl (conj C (conj S B))).
  - discriminate.
Qed.

Lemma sets_inv_run (n : nat) :
  ∀ s s' : state, sets_inv s -> run n s = Step s' -> sets_inv s'.
Proof.
  induction n as [ | n IH]; intros s s' Hinv; simpl.
  - intros [= <-]. exact Hinv.
  - destruct (step s) as [next | k vs | ] eqn:Hstep; try discriminate.
    exact (IH next s' (sets_inv_step s next Hinv Hstep)).
Qed.

Theorem sets_inv_initial : sets_inv_initial_statement.
Proof.
  intros e Hsets. unfold sets_inv. simpl_state.
  refine (conj Hsets (conj I (conj _ _))).
  - intros id c [=].
  - intros sym slot symbols decls slot_values [=].
Qed.

Theorem well_formed_sets_inv : well_formed_sets_inv_statement.
Proof.
  intros u e n s Hwf.
  apply sets_inv_run. apply sets_inv_initial.
  exact (well_formed_unit_sets_well_formed u e Hwf).
Qed.

Theorem well_formed_makes_closures : well_formed_makes_closures_statement.
Proof.
  intros u e n s Hwf Hrun.
  destruct (well_formed_sets_inv u e n s Hwf Hrun) as (P & _).
  unfold makes_closure_outside_set.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    try reflexivity.
  - simpl in P. apply andb_true_iff in P as [Pd _].
    destruct d as [smp | p args | code slots | fslot decls slots
      | slot closure | pfrom slot closure | move_from move_to closure];
      try reflexivity.
    simpl in Pd. unfold set_well_formed in Pd.
    apply andb_true_iff in Pd as [Pd _].
    apply andb_true_iff in Pd as [-> _]. reflexivity.
  - simpl in P. apply andb_true_iff in P as [P _].
    apply andb_true_iff in P as [P _].
    apply andb_true_iff in P as [Pwhole _].
    match goal with
    | |- existsb ?bad bindings = false =>
      destruct (existsb bad bindings) eqn:Hexists; [ | reflexivity ]
    end.
    apply existsb_exists in Hexists as [binding [Hin Hbad]].
    rewrite forallb_forall in Pwhole. specialize (Pwhole binding Hin).
    destruct binding
      as [id c | id
         | sym
             [code slots | fslot syms decls slots | tag fields | contents
             | bytes | elements]];
      try discriminate.
    destruct (static_set_whole_facts bindings sym fslot syms decls slots
                Pwhole) as (Hslot & _).
    rewrite Hslot in Hbad. discriminate.
Qed.

Theorem well_formed_sets_bound : well_formed_sets_bound_statement.
Proof.
  intros u e n s Hwf Hrun.
  destruct (well_formed_sets_inv u e n s Hwf Hrun) as (_ & _ & _ & B).
  exact B.
Qed.

Theorem project_function_slot_bound : project_function_slot_bound_statement.
Proof.
  intros statics move_from move_to v w Hbound.
  unfold project_function_slot.
  destruct v as [c | code slots | slot decls slots | s | tag fields
    | | contents | l | pa_callee pa_args pa_arity];
    [ | | | | discriminate .. ].
  - discriminate.
  - intros Hproject.
    apply project_unnamed_function_slot in Hproject as (-> & -> & ->).
    exists [(unnamed_function_slot, code)], slots. auto.
  - destruct (Pos.eqb_spec slot move_from) as [-> | _]; [ | discriminate ].
    destruct (lookup move_from decls) as [code | ] eqn:Hfrom;
      [ | discriminate ].
    destruct (lookup move_to decls) as [code' | ] eqn:Hto; [ | discriminate ].
    simpl. intros [= <-]. exists decls, slots. simpl. rewrite Hfrom, Hto.
    auto.
  - destruct (lookup s (static_values statics))
      as [[code slots | slot syms decls slots | tag fields | contents
          | bytes | elements | ] | ] eqn:Hs;
      try discriminate.
    + intros Hproject.
      apply project_unnamed_function_slot in Hproject as (-> & -> & ->).
      exists [(unnamed_function_slot, code)], slots. simpl. rewrite Hs. auto.
    + destruct (Pos.eqb_spec slot move_from) as [-> | _]; [ | discriminate ].
      destruct (lookup move_from decls) as [code | ] eqn:Hfrom;
        [ | discriminate ].
      destruct (lookup move_to decls) as [code' | ] eqn:Hto;
        [ | discriminate ].
      destruct (lookup move_to syms) as [s' | ] eqn:Hsymbol;
        [ | discriminate ].
      simpl. intros [= <-]. exists decls, slots.
      destruct (Hbound s move_from syms decls slots Hs) as [_ Hset].
      pose proof (Hset move_to s' (lookup_in _ _ _ Hsymbol)) as Hs'.
      simpl. rewrite Hs, Hs', Hfrom, Hto. auto.
Qed.

(* ======================================================================== *)
(* Valid values                                                             *)
(* ======================================================================== *)

Lemma slot_values_valid_in (slot_values : list (value_slot * value)) :
  forallb (fun '(_, contents) => value_valid contents) slot_values = true ->
  ∀ (w : value_slot) (contents : value),
    (w, contents) ∈ slot_values -> value_valid contents = true.
Proof.
  intros H w contents Hin. rewrite forallb_forall in H.
  exact (H (w, contents) Hin).
Qed.

Lemma statics_valid_lookup (statics : static_env) (s : symbol)
    (allocated : statically_allocated_value) :
  statics_valid statics ->
  lookup s (static_values statics) = Some allocated ->
  allocated_valid allocated = true.
Proof.
  intros Hstatics Hs. unfold statics_valid in Hstatics.
  rewrite Forall_forall in Hstatics.
  exact (Hstatics (s, allocated) (lookup_in _ _ _ Hs)).
Qed.

Lemma unnamed_set_well_formed (code : code_id) :
  set_well_formed unnamed_function_slot [(unnamed_function_slot, code)]
  = true.
Proof. reflexivity. Qed.

(* A map has the names of another: what is bound in one is in the other. *)
Lemma is_bound_same_names {A B : Type} (name : positive)
    (bindings : list (positive * A)) (bindings' : list (positive * B)) :
  map fst bindings = map fst bindings' ->
  is_bound name bindings' = true -> is_bound name bindings = true.
Proof.
  intros Hnames. rewrite !is_bound_among, Hnames. exact (fun H => H).
Qed.

Theorem closure_in_set_valid : closure_in_set_valid_statement.
Proof.
  intros statics v slot decls slot_values Hstatics Hv.
  unfold closure_in_set.
  destruct v as [c | code slots | at_slot set_decls slots | s | tag fields
    | | contents | l | pa_callee pa_args pa_arity];
    [ | | | | discriminate .. ].
  - discriminate.
  - intros [= <- <- <-]. split; [ exact (unnamed_set_well_formed code) | ].
    exact (slot_values_valid_in slots Hv).
  - simpl in Hv. apply andb_true_iff in Hv as [Hset Hslots].
    destruct (lookup at_slot set_decls); [ | discriminate ].
    simpl. intros [= <- <- <-]. split; [ exact Hset | ].
    exact (slot_values_valid_in slots Hslots).
  - destruct (lookup s (static_values statics))
      as [[code slots | at_slot syms set_decls slots | tag fields | contents
          | bytes | elements | ] | ] eqn:Hs;
      try discriminate.
    + pose proof (statics_valid_lookup statics s _ Hstatics Hs) as Hvalid.
      intros [= <- <- <-]. split; [ exact (unnamed_set_well_formed code) | ].
      exact (slot_values_valid_in slots Hvalid).
    + pose proof (statics_valid_lookup statics s _ Hstatics Hs) as Hvalid.
      simpl in Hvalid. apply andb_true_iff in Hvalid as [Hvalid Hslots].
      apply andb_true_iff in Hvalid as [Hset _].
      destruct (lookup at_slot set_decls); [ | discriminate ].
      simpl. intros [= <- <- <-]. split; [ exact Hset | ].
      exact (slot_values_valid_in slots Hslots).
Qed.

(* The projection of a function slot from a closure of a set of one
   function, with no slot named, where the set has the slot asked for. *)
Lemma project_unnamed_defined (move_to : function_slot) (code : code_id)
    (v : value) :
  is_bound move_to [(unnamed_function_slot, code)] = true ->
  ∃ w : value,
    (if Pos.eqb unnamed_function_slot unnamed_function_slot
     then if Pos.eqb move_to unnamed_function_slot then Some v else None
     else None)
    = Some w.
Proof.
  unfold is_bound. cbn [lookup].
  destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
  intros _. exists v. reflexivity.
Qed.

Theorem project_function_slot_valid : project_function_slot_valid_statement.
Proof.
  intros statics move_from move_to v decls slot_values Hstatics.
  unfold closure_in_set, project_function_slot.
  destruct v as [c | code slots | at_slot set_decls slots | s | tag fields
    | | contents | l | pa_callee pa_args pa_arity];
    [ | | | | discriminate .. ].
  - discriminate.
  - intros [= <- <- <-]. exact (project_unnamed_defined move_to code _).
  - destruct (lookup at_slot set_decls) as [code | ]; [ | discriminate ].
    simpl. intros [= <- <- <-]. unfold is_bound.
    destruct (lookup move_to set_decls) as [code' | ]; [ | discriminate ].
    intros _. eexists. rewrite Pos.eqb_refl. simpl. reflexivity.
  - destruct (lookup s (static_values statics))
      as [[code slots | at_slot syms set_decls slots | tag fields | contents
          | bytes | elements | ] | ] eqn:Hs;
      try discriminate.
    + intros [= <- <- <-]. exact (project_unnamed_defined move_to code _).
    + pose proof (statics_valid_lookup statics s _ Hstatics Hs) as Hvalid.
      simpl in Hvalid. apply andb_true_iff in Hvalid as [Hvalid _].
      apply andb_true_iff in Hvalid as [_ Hnames].
      destruct (list_eq_dec Pos.eq_dec (map fst syms) (map fst set_decls))
        as [Hsame | ]; [ | discriminate ].
      destruct (lookup at_slot set_decls) as [code | ]; [ | discriminate ].
      simpl. intros [= <- <- <-] Hto.
      pose proof (is_bound_same_names move_to syms set_decls Hsame Hto)
        as Hsymbol.
      unfold is_bound in Hto, Hsymbol.
      destruct (lookup move_to set_decls) as [code' | ]; [ | discriminate ].
      destruct (lookup move_to syms) as [s' | ]; [ | discriminate ].
      eexists. rewrite Pos.eqb_refl. simpl. reflexivity.
Qed.

Lemma eval_simple_valid (vars : env) (s : simple) (v : value) :
  env_valid vars -> eval_simple vars s = Some v -> value_valid v = true.
Proof.
  intros Hvars. destruct s as [x | sym | c]; simpl.
  - exact (env_values_all_lookup _ vars x v Hvars).
  - intros [= <-]. reflexivity.
  - intros [= <-]. reflexivity.
Qed.

Lemma eval_simples_valid (vars : env) :
  env_valid vars ->
  ∀ (args : list simple) (vs : list value),
    eval_simples vars args = Some vs -> forallb value_valid vs = true.
Proof.
  intros Hvars. induction args as [ | s args IH]; intros vs; simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_valid vars s v Hvars Hs). exact (IH rest eq_refl).
Qed.

Lemma eval_value_slots_valid (vars : env) :
  env_valid vars ->
  ∀ (value_slots : list (value_slot * simple))
    (slot_values : list (value_slot * value)),
    eval_value_slots vars value_slots = Some slot_values ->
    forallb (fun '(_, contents) => value_valid contents) slot_values = true.
Proof.
  intros Hvars.
  induction value_slots as [ | [slot s] value_slots IH]; intros slot_values;
    simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_value_slots vars value_slots) as [rest | ];
      [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_valid vars s v Hvars Hs). exact (IH rest eq_refl).
Qed.

(* Validity is a property of values of the kind that a primitive keeps
   (FlambdaPrimitiveSpec.eval_prim_preserves_statement). *)
Lemma value_valid_of_numbers :
  holds_of_numbers (fun v => value_valid v = true).
Proof. intros c. split; reflexivity. Qed.

Lemma value_valid_of_blocks : holds_of_blocks (fun v => value_valid v = true).
Proof. intros tag fields. exact (forallb_forall value_valid fields). Qed.

Lemma value_valid_of_arrays : holds_of_arrays (fun v => value_valid v = true).
Proof. intros elements. exact (forallb_forall value_valid elements). Qed.

Lemma value_valid_in_static_blocks (statics : static_env) :
  statics_valid statics ->
  holds_in_static_blocks statics (fun v => value_valid v = true).
Proof.
  intros Hstatics s tag fields _ Hs v Hin.
  pose proof (statics_valid_lookup statics s _ Hstatics Hs) as Hvalid.
  simpl in Hvalid. rewrite forallb_forall in Hvalid. exact (Hvalid v Hin).
Qed.

Lemma value_valid_in_static_arrays (statics : static_env) :
  statics_valid statics ->
  holds_in_static_arrays statics (fun v => value_valid v = true).
Proof.
  intros Hstatics s elements _ Hs v Hin.
  pose proof (statics_valid_lookup statics s _ Hstatics Hs) as Hvalid.
  simpl in Hvalid. rewrite forallb_forall in Hvalid. exact (Hvalid v Hin).
Qed.

(* The closure at another slot of the set of a valid closure is valid. *)
Lemma project_function_slot_keeps_valid (statics : static_env)
    (move_from move_to : function_slot) (v w : value) :
  value_valid v = true ->
  project_function_slot statics move_from move_to v = Some w ->
  value_valid w = true.
Proof.
  intros Hv. unfold project_function_slot.
  destruct v as [c | code slots | at_slot decls slots | s | tag fields
    | | contents | l | pa_callee pa_args pa_arity];
    [ | | | | discriminate .. ].
  - discriminate.
  - intros Hproject.
    apply project_unnamed_function_slot in Hproject as (_ & _ & ->).
    exact Hv.
  - destruct (Pos.eqb at_slot move_from); [ | discriminate ].
    destruct (lookup at_slot decls); [ | discriminate ].
    destruct (lookup move_to decls) as [code | ] eqn:Hto; [ | discriminate ].
    simpl. intros [= <-]. simpl in Hv. simpl.
    apply andb_true_iff in Hv as [Hset Hslots]. rewrite Hslots.
    unfold set_well_formed in Hset. unfold set_well_formed.
    apply andb_true_iff in Hset as [Hset Hcodes].
    apply andb_true_iff in Hset as [_ Hnames].
    unfold is_bound. rewrite Hto, Hnames, Hcodes. reflexivity.
  - destruct (lookup s (static_values statics))
      as [[code slots | at_slot syms decls slots | tag fields | contents
          | bytes | elements | ] | ];
      try discriminate.
    + intros Hproject.
      apply project_unnamed_function_slot in Hproject as (_ & _ & ->).
      reflexivity.
    + destruct (Pos.eqb at_slot move_from); [ | discriminate ].
      destruct (lookup at_slot decls); [ | discriminate ].
      destruct (lookup move_to decls); [ | discriminate ].
      destruct (lookup move_to syms); [ | discriminate ].
      simpl. intros [= <-]. reflexivity.
Qed.

Theorem eval_named_valid : eval_named_valid_statement.
Proof.
  intros statics vars d v Hd Hvars Hstatics.
  destruct d as [s | p args | code slots | slot decls slots | slot closure
    | project_from slot closure | move_from move_to closure];
    simpl.
  - exact (eval_simple_valid vars s v Hvars).
  - destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
      [ | discriminate ].
    simpl. intros Hprim.
    apply (eval_prim_preserves (fun v => value_valid v = true) statics p vs
             v value_valid_of_numbers value_valid_of_blocks
             value_valid_of_arrays
             (value_valid_in_static_blocks statics Hstatics)
             (value_valid_in_static_arrays statics Hstatics));
      [ | exact Hprim ].
    pose proof (eval_simples_valid vars Hvars args vs Hargs) as Hvs.
    rewrite forallb_forall in Hvs. exact Hvs.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_value_slots_valid vars Hvars slots vs Hslots).
  - destruct (lookup slot decls); [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. simpl. simpl in Hd. rewrite Hd.
    exact (eval_value_slots_valid vars Hvars slots vs Hslots).
  - destruct (eval_simple vars closure) as [c | ] eqn:Hc; [ | discriminate ].
    simpl.
    destruct (closure_of statics c) as [[code slot_values] | ] eqn:Hclosure;
      [ | discriminate ].
    simpl. intros Hlookup.
    destruct (proj1 (closure_in_set_of statics c code slot_values) Hclosure)
      as (at_slot & decls & Hin_set & _).
    destruct (closure_in_set_valid statics c at_slot decls slot_values
                Hstatics (eval_simple_valid vars closure c Hvars Hc) Hin_set)
      as [_ Hslots].
    exact (Hslots slot v (lookup_in _ _ _ Hlookup)).
  - destruct (eval_simple vars closure) as [c | ] eqn:Hc; [ | discriminate ].
    simpl. unfold project_value_slot.
    destruct (closure_in_set statics c) as [[[at_slot decls] slot_values] | ]
      eqn:Hin_set;
      [ | discriminate ].
    simpl. destruct (Pos.eqb at_slot project_from); [ | discriminate ].
    intros Hlookup.
    destruct (closure_in_set_valid statics c at_slot decls slot_values
                Hstatics (eval_simple_valid vars closure c Hvars Hc) Hin_set)
      as [_ Hslots].
    exact (Hslots slot v (lookup_in _ _ _ Hlookup)).
  - destruct (eval_simple vars closure) as [c | ] eqn:Hc; [ | discriminate ].
    simpl.
    exact (project_function_slot_keeps_valid statics move_from move_to c v
             (eval_simple_valid vars closure c Hvars Hc)).
Qed.

Theorem eval_static_const_valid : eval_static_const_valid_statement.
Proof.
  intros bindings s const vars allocated Hwhole Hvars.
  destruct const as [code slots | slot syms decls slots | tag fields
    | contents | bytes | elements];
    simpl.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_value_slots_valid vars Hvars slots vs Hslots).
  - simpl in Hwhole. apply andb_true_iff in Hwhole as [Hwhole _].
    apply andb_true_iff in Hwhole as [Hwhole _].
    apply andb_true_iff in Hwhole as [Hset Hnames].
    destruct (lookup slot decls); [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. simpl. rewrite Hset, Hnames.
    exact (eval_value_slots_valid vars Hvars slots vs Hslots).
  - destruct (eval_simples vars fields) as [vs | ] eqn:Hfields;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simples_valid vars Hvars fields vs Hfields).
  - destruct (eval_simple vars contents) as [v | ] eqn:Hcontents;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simple_valid vars contents v Hvars Hcontents).
  - intros [= <-]. reflexivity.
  - destruct (eval_simples vars elements) as [vs | ] eqn:Helements;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simples_valid vars Hvars elements vs Helements).
Qed.

(* ======================================================================== *)
(* The values of a well-formed unit                                         *)
(* ======================================================================== *)

(* A group whose bindings pass the check of static sets keeps what the
   symbols are bound to valid. [all] is the whole group. *)
Lemma eval_static_bindings_valid (vars : env) (all : list static_binding) :
  env_valid vars ->
  ∀ (bindings : list static_binding) (statics statics' : static_env),
    forallb (static_set_whole all) bindings = true ->
    statics_valid statics ->
    eval_static_bindings vars bindings statics = Some statics' ->
    statics_valid statics'.
Proof.
  intros Hvars.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics statics'; simpl.
  - intros _ Hvalid [= <-]. exact Hvalid.
  - intros Hrest Hvalid.
    destruct (lookup id (static_code statics)); [ discriminate | ].
    apply (IH _ _ Hrest). exact Hvalid.
  - intros Hrest Hvalid. exact (IH _ _ Hrest Hvalid).
  - intros H Hvalid. apply andb_true_iff in H as [Hwhole Hrest].
    destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ] eqn:Hconst;
      simpl; [ | discriminate ].
    apply (IH _ _ Hrest). unfold statics_valid. simpl.
    apply Forall_cons; [ | exact Hvalid ].
    exact (eval_static_const_valid all sym const vars allocated Hwhole Hvars
             Hconst).
Qed.

Lemma values_valid_Forall (vs : list value) :
  forallb value_valid vs = true -> Forall (fun v => value_valid v = true) vs.
Proof.
  intros H. rewrite forallb_forall in H. apply Forall_forall. exact H.
Qed.

(* A partial application is valid when the value it calls and its
   arguments are. *)
Lemma value_valid_partial (callee : value) (args : list value)
    (args_arity : arity) :
  value_valid (Value_partial_application callee args args_arity) = true
  <-> value_valid callee = true ∧ Forall (fun v => value_valid v = true) args.
Proof. simpl. rewrite andb_true_iff, forallb_Forall. reflexivity. Qed.

(* A step from a state whose terms have well-formed sets keeps the values
   valid: the four things that FlambdaSemanticsSpec.step_values_all_statement
   asks of the state. *)
Theorem state_valid_step : state_valid_step_statement.
Proof.
  intros s (P & _) Hvalid.
  apply (step_values_all (fun _ v => value_valid v = true)
           (fun _ allocated => allocated_valid allocated = true) s
           (fun _ => value_valid_partial) Hvalid).
  - intros size tuple fields Htuple Hfields.
    destruct Hvalid as (_ & _ & Hstatics & _).
    apply (tuple_fields_forall _ (state_statics s) size tuple fields);
      [ | exact Hfields ].
    intros field v Hprim.
    refine (eval_prim_preserves (fun v => value_valid v = true)
              (state_statics s) _ [tuple] v value_valid_of_numbers
              value_valid_of_blocks value_valid_of_arrays
              (value_valid_in_static_blocks (state_statics s) Hstatics)
              (value_valid_in_static_arrays (state_statics s) Hstatics) _
              Hprim).
    intros w [<- | []]. exact Htuple.
  - intros x kind d body v hp' rd' Hexpr Hexec. rewrite Hexpr in P. simpl in P.
    apply andb_true_iff in P as [Pd _].
    destruct Hvalid as (Hvars & _ & Hstatics & Hheap).
    refine (exec_named_values_all (fun v => value_valid v = true)
              (state_statics s) (state_heap s) hp' (state_reads s) rd'
                (state_vars s) d v
              value_valid_of_numbers value_valid_of_blocks
              value_valid_of_arrays
              (value_valid_in_static_blocks (state_statics s) Hstatics)
              (value_valid_in_static_arrays (state_statics s) Hstatics)
              (fun _ => eq_refl) Hheap _ _ Hexec).
    + intros p args vs _ Hvs.
      exact (proj1 (forallb_forall _ _)
               (eval_simples_valid (state_vars s) Hvars args vs Hvs)).
    + intros _ w Heval.
      exact (eval_named_valid (state_statics s) (state_vars s) d w Pd Hvars
               Hstatics Heval).
  - intros bindings body statics' Hexpr Heval.
    split; [ exact (fun v H => H) | ].
    rewrite Hexpr in P. simpl in P.
    apply andb_true_iff in P as [P _].
    apply andb_true_iff in P as [P _].
    apply andb_true_iff in P as [Pwhole _].
    destruct Hvalid as (Hvars & _ & Hstatics & _).
    exact (eval_static_bindings_valid (state_vars s) bindings Hvars bindings
             (state_statics s) statics' Pwhole Hstatics Heval).
  - intros ap callee vs extra _ Hcallee Hargs Hextra.
    destruct Hvalid as (Hvars & _).
    split; [ | split ].
    + exact (eval_simple_valid (state_vars s) (ap_callee ap) callee Hvars
               Hcallee).
    + exact (values_valid_Forall vs
               (eval_simples_valid (state_vars s) Hvars (ap_args ap) vs
                  Hargs)).
    + exact (values_valid_Forall extra
               (eval_simples_valid (state_vars s) Hvars
                  (exn_extra_simples ap) extra Hextra)).
  - intros ac vs _ Hargs. destruct Hvalid as (Hvars & _).
    exact (values_valid_Forall vs
             (eval_simples_valid (state_vars s) Hvars (ac_args ac) vs Hargs)).
Qed.

Lemma state_valid_run (n : nat) :
  ∀ s : state,
    sets_inv s -> state_valid s ->
    match run n s with
    | Step s' => state_valid s'
    | Exit _ vs _ _ => Forall (fun v => value_valid v = true) vs
    | Undefined => True
    end.
Proof.
  induction n as [ | n IH]; intros s Hsets Hvalid; simpl.
  - exact Hvalid.
  - pose proof (state_valid_step s Hsets Hvalid) as Hstep.
    destruct (step s) as [next | k vs | ] eqn:Hnext.
    + exact (IH next (sets_inv_step s next Hsets Hnext) Hstep).
    + exact Hstep.
    + exact I.
Qed.

(* The state a unit starts in holds no value. *)
Lemma initial_state_valid (e : expr) :
  state_valid (State e [] [] empty_static_env empty_heap no_reads).
Proof.
  exact
    (conj (Forall_nil _)
       (conj I (conj (Forall_nil _) (heap_values_all_empty _)))).
Qed.

Theorem well_formed_values_valid : well_formed_values_valid_statement.
Proof.
  intros u e n s Hwf Hrun.
  pose proof
    (state_valid_run n (State e [] [] empty_static_env empty_heap no_reads)
       (well_formed_sets_inv u e 0 _ Hwf eq_refl)
       (initial_state_valid e))
    as H.
  rewrite Hrun in H. exact H.
Qed.

Theorem well_formed_exit_valid : well_formed_exit_valid_statement.
Proof.
  intros u e n k vs hp rd Hwf Hrun.
  pose proof
    (state_valid_run n (State e [] [] empty_static_env empty_heap no_reads)
       (well_formed_sets_inv u e 0 _ Hwf eq_refl)
       (initial_state_valid e))
    as H.
  rewrite Hrun in H. rewrite Forall_forall in H. exact H.
Qed.

Theorem well_formed_closure_slot : well_formed_closure_slot_statement.
Proof.
  intros u e n s closure v slot slot' decls slot_values slot_values'
    code Hwf Hrun Hv Hin_set Hclosure Hslot'.
  destruct (well_formed_values_valid u e n s Hwf Hrun)
    as (Hvars & _ & Hstatics & _).
  destruct (closure_in_set_valid (state_statics s) v slot decls slot_values
              Hstatics (eval_simple_valid (state_vars s) closure v Hvars Hv)
              Hin_set)
    as [Hset _].
  unfold set_well_formed in Hset. apply andb_true_iff in Hset as [_ Hcodes].
  exact (closure_code_determines_slot (state_statics s) v slot slot' decls
           slot_values slot_values' code Hin_set Hcodes Hclosure Hslot').
Qed.

(* A variable that is not a parameter is bound after a call to what it
   was bound to before the parameters. *)
Lemma bind_params_lookup_other (x : variable) (params : list variable) :
  ∀ (vs : list value) (vars vars' : env),
    among x params = false ->
    bind_params params vs vars = Some vars' ->
    lookup x vars' = lookup x vars.
Proof.
  unfold among.
  induction params as [ | y params IH]; intros [ | v vs] vars vars'; simpl;
    try discriminate.
  - intros _ [= <-]. reflexivity.
  - intros Hx Hbind. apply orb_false_iff in Hx as [Hxy Hrest].
    rewrite (IH vs _ vars' Hrest Hbind). simpl. rewrite Hxy. reflexivity.
Qed.

Theorem call_binds_my_closure : call_binds_my_closure_statement.
Proof.
  intros callee kind vs args_arity return_arity return_to exn_to statics hp rd s
    called slot_values c Hcall Hclosure Hcode Hfresh Hcount.
  unfold call_function in Hcall.
  rewrite (shape_of_call_closure statics callee _ _ _ _
             (closure_not_partial statics callee _ Hclosure))
    in Hcall.
  destruct
    (shape_of_closure_call statics callee kind vs args_arity return_arity)
    as [shape | ] eqn:Hshape;
    [ | discriminate ].
  apply shape_of_closure_call_cases in Hshape
    as (called' & slots' & c' & vs' & args_arity' & Hclosure' & Hcode' & _
        & Hargs & Hshape).
  rewrite Hclosure in Hclosure'. injection Hclosure' as <- <-.
  rewrite Hcode in Hcode'. injection Hcode' as <-.
  assert (Hbody :
    ∀ (args : list value) (at_arity result : arity) (k : cont_value),
      call_code callee c args at_arity result k exn_to statics hp rd = Step s ->
      state_expr s = code_body c
      ∧ lookup (code_my_closure c) (state_vars s) = Some callee).
  { intros args at_arity result k Hstep.
    apply call_code_step in Hstep
      as (callee_vars & _ & _ & _ & _ & Hbind & ->).
    simpl. split; [ reflexivity | ].
    rewrite (bind_params_lookup_other _ _ args _ callee_vars Hfresh Hbind).
    simpl. rewrite Pos.eqb_refl. reflexivity. }
  apply shape_of_code_call_cases in Hshape
    as [[_ ->] | [(_ & _ & _ & ->) | (Hfewer & _ & _ & Htupled & ->)]];
    cbn [run_call] in Hcall.
  - exact (Hbody _ _ _ _ Hcall).
  - exact (Hbody _ _ _ _ Hcall).
  - exfalso.
    rewrite (arguments_for_code_not_tupled _ _ _ _ Htupled) in Hargs.
    injection Hargs as <- <-.
    exact (Nat.lt_irrefl _ (Nat.lt_le_trans _ _ _ Hfewer Hcount)).
Qed.

(* ======================================================================== *)
(* A test that the terms of a state keep passing                            *)
(* ======================================================================== *)

(* A test on expressions, on handlers and on code that goes down into
   the parts of a term: every state of a run holds terms that pass it,
   if the first does. The terms of a state are its expression, the
   handlers that its continuations keep, and the code that is bound. *)
Section Terms_kept.

Variable Pe : expr -> bool.
Variable Ph : continuation_handler -> bool.
Variable Pc : code -> bool.

Hypothesis Pe_let :
  ∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr),
    Pe (Let x kind d body) = true -> Pe body = true.
Hypothesis Pe_let_static :
  ∀ (bindings : list static_binding) (body : expr),
    Pe (Let_static bindings body) = true ->
    Pe body = true
    ∧ ∀ (id : code_id) (c : code), Bind_code id c ∈ bindings -> Pc c = true.
Hypothesis Pe_let_cont :
  ∀ (k : continuation) (h : continuation_handler) (body : expr),
    Pe (Let_cont_non_recursive k h body) = true ->
    Ph h = true ∧ Pe body = true.
Hypothesis Pe_let_cont_rec :
  ∀ (handlers : list (continuation * continuation_handler)) (body : expr),
    Pe (Let_cont_recursive handlers body) = true ->
    forallb (fun '(_, h) => Ph h) handlers = true ∧ Pe body = true.
Hypothesis Pe_apply_cont : ∀ ac : apply_cont, Pe (Apply_cont ac) = true.
Hypothesis Ph_handler :
  ∀ (params : list variable) (metadata : handler_metadata) (handler : expr),
    Ph (Handler params metadata handler) = true -> Pe handler = true.
Hypothesis Pc_body : ∀ c : code, Pc c = true -> Pe (code_body c) = true.

Fixpoint cont_terms_ok (c : cont_value) : Prop :=
  match c with
  | Outer _ => True
  | Closure h _ conts recursive =>
    Ph h = true
    ∧ forallb (fun '(_, h) => Ph h) recursive = true
    ∧ fold_right (fun kc acc => cont_terms_ok (snd kc) ∧ acc) True conts
  | Extra_args c' _ => cont_terms_ok c'
  | Over_application _ _ _ return_to exn_to =>
    cont_terms_ok return_to ∧ cont_terms_ok exn_to
  end.

Definition conts_terms_ok (conts : cont_env) : Prop :=
  fold_right (fun kc acc => cont_terms_ok (snd kc) ∧ acc) True conts.

Lemma conts_terms_ok_app (conts conts' : cont_env) :
  conts_terms_ok conts -> conts_terms_ok conts' ->
  conts_terms_ok (conts ++ conts').
Proof.
  unfold conts_terms_ok.
  induction conts as [ | kc conts IH]; simpl; [ auto | ].
  intros [Hc Hrest] H'. split; [ exact Hc | exact (IH Hrest H') ].
Qed.

Lemma resolve_terms_ok (k : continuation) (conts : cont_env) :
  conts_terms_ok conts -> cont_terms_ok (resolve k conts).
Proof.
  unfold resolve, conts_terms_ok.
  induction conts as [ | [k' c] conts IH]; simpl; [ intros _; exact I | ].
  intros [Hc Hrest]. destruct (Pos.eqb k k'); [ exact Hc | exact (IH Hrest) ].
Qed.

Lemma rec_closures_terms_ok (vars : env) (conts : cont_env)
    (all handlers : list (continuation * continuation_handler)) :
  forallb (fun '(_, h) => Ph h) all = true ->
  conts_terms_ok conts ->
  forallb (fun '(_, h) => Ph h) handlers = true ->
  conts_terms_ok (map (fun '(k, h) => (k, Closure h vars conts all)) handlers).
Proof.
  intros Hall Hconts. unfold conts_terms_ok.
  induction handlers as [ | [k h] handlers IH]; simpl;
    [ intros _; exact I | ].
  intros H. apply andb_true_iff in H as [Hh Hrest].
  split; [ | exact (IH Hrest) ].
  split; [ exact Hh | ]. split; [ exact Hall | exact Hconts ].
Qed.

Lemma with_extra_args_terms_ok (c : cont_value) (extra : list value) :
  cont_terms_ok c -> cont_terms_ok (with_extra_args c extra).
Proof. destruct extra; intros Hc; exact Hc. Qed.

Definition code_terms_ok (statics : static_env) : Prop :=
  ∀ (id : code_id) (c : code),
    lookup id (static_code statics) = Some c -> Pc c = true.

Lemma eval_static_bindings_code_terms_ok (vars : env)
    (bindings : list static_binding) :
  ∀ statics statics' : static_env,
    (∀ (id : code_id) (c : code), Bind_code id c ∈ bindings -> Pc c = true) ->
    code_terms_ok statics ->
    eval_static_bindings vars bindings statics = Some statics' ->
    code_terms_ok statics'.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics statics'; simpl.
  - intros _ Hok [= <-]. exact Hok.
  - intros H Hok.
    destruct (lookup id (static_code statics)); [ discriminate | ].
    apply IH; [ intros id' c' Hin; apply (H id' c'); right; exact Hin | ].
    intros id' c'. simpl.
    destruct (Pos.eqb id' id);
      [ intros [= <-]; apply (H id c); left; reflexivity | apply Hok ].
  - intros H Hok.
    apply IH; [ intros id' c' Hin; apply (H id' c'); right; exact Hin | ].
    exact Hok.
  - intros H Hok.
    destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const); simpl; [ | discriminate ].
    apply IH; [ intros id' c' Hin; apply (H id' c'); right; exact Hin | ].
    exact Hok.
Qed.

Definition terms_inv (s : state) : Prop :=
  Pe (state_expr s) = true
  ∧ conts_terms_ok (state_conts s)
  ∧ code_terms_ok (state_statics s).

(* Where a call and the application of a continuation land
   (FlambdaSemanticsProofs.call_function_lands). *)
Lemma terms_inv_over (remaining : list value)
    (remaining_arity return_arity : arity) (return_to exn_to : cont_value) :
  cont_terms_ok
    (Over_application remaining remaining_arity return_arity return_to
       exn_to)
  <-> cont_terms_ok return_to ∧ cont_terms_ok exn_to.
Proof. reflexivity. Qed.

Lemma terms_inv_code (statics : static_env) (hp : heap) (rd : read_record) :
  code_terms_ok statics ->
  ∀ (called : code_id) (c : code) (callee_vars : env)
    (return_to exn_to : cont_value),
    lookup called (static_code statics) = Some c ->
    cont_terms_ok return_to ->
    cont_terms_ok exn_to ->
    terms_inv
      (State (code_body c) callee_vars
         [(code_return_continuation c, return_to);
          (code_exn_continuation c, exn_to)]
         statics hp rd).
Proof.
  intros S called c callee_vars return_to exn_to Hlookup Hreturn Hexn.
  unfold terms_inv. simpl_state.
  exact (conj (Pc_body c (S called c Hlookup))
           (conj (conj Hreturn (conj Hexn I)) S)).
Qed.

Lemma terms_inv_handler (statics : static_env) (hp : heap) (rd : read_record) :
  code_terms_ok statics ->
  ∀ (params : list variable) (metadata : handler_metadata) (handler : expr)
    (def_vars : env) (def_conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (handler_vars : env),
    cont_terms_ok
      (Closure (Handler params metadata handler) def_vars def_conts
         recursive) ->
    terms_inv
      (State handler handler_vars
         (rec_closures recursive def_vars def_conts ++ def_conts) statics hp
           rd).
Proof.
  intros S params metadata handler def_vars def_conts recursive handler_vars
    Cr.
  unfold terms_inv. simpl_state. simpl in Cr.
  destruct Cr as (Hh & Hrec & Cdef).
  refine (conj (Ph_handler params metadata handler Hh)
            (conj (conts_terms_ok_app _ _ _ Cdef) S)).
  exact (rec_closures_terms_ok def_vars def_conts recursive recursive Hrec
           Cdef Hrec).
Qed.

Lemma terms_inv_step (s s' : state) :
  terms_inv s -> step s = Step s' -> terms_inv s'.
Proof.
  intros (P & C & S) Hstep. rewrite (state_eta s) in Hstep.
  set (vars := state_vars s) in *. set (conts := state_conts s) in *.
  set (statics := state_statics s) in *.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step in Hstep.
  - (* Let *)
    destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
    injection Hstep as <-. unfold terms_inv. simpl_state.
    exact (conj (Pe_let x kind d body P) (conj C S)).
  - (* Let_static *)
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Heval; [ | discriminate ].
    injection Hstep as <-. unfold terms_inv. simpl_state.
    destruct (Pe_let_static bindings body P) as [Pbody Pcode].
    refine (conj Pbody (conj C _)).
    exact (eval_static_bindings_code_terms_ok vars bindings statics statics'
             Pcode S Heval).
  - (* Let_cont_non_recursive *)
    injection Hstep as <-. unfold terms_inv. simpl_state.
    destruct (Pe_let_cont k h body P) as [Hh Pbody].
    refine (conj Pbody (conj (conj _ C) S)).
    simpl. exact (conj Hh (conj eq_refl C)).
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers); [ discriminate | ].
    injection Hstep as <-. unfold terms_inv. simpl_state.
    destruct (Pe_let_cont_rec handlers body P) as [Phandlers Pbody].
    refine (conj Pbody (conj (conts_terms_ok_app _ _ _ C) S)).
    exact (rec_closures_terms_ok vars conts handlers handlers Phandlers C
             Phandlers).
  - (* Apply *)
    destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | discriminate ].
    refine
      (call_function_lands statics _ rd_ cont_terms_ok terms_inv
         (fun c extra H => H) terms_inv_over (terms_inv_code statics _ rd_ S)
         (terms_inv_handler statics _ rd_ S) _ _ _ _ _ _ _ s'
         (resolve_terms_ok _ conts C) _ Hstep).
    apply with_extra_args_terms_ok, resolve_terms_ok, C.
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    unfold apply_continuation in Hstep.
    exact
      (apply_cont_value_lands statics _ rd_ cont_terms_ok terms_inv
         (fun c extra H => H) terms_inv_over (terms_inv_code statics _ rd_ S)
         (terms_inv_handler statics _ rd_ S) _ vs s'
           (resolve_terms_ok _ conts C)
         Hstep).
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ]; [ | discriminate ].
    injection Hstep as <-. unfold terms_inv. simpl_state.
    exact (conj (Pe_apply_cont ac) (conj C S)).
  - discriminate.
Qed.

Lemma terms_inv_initial (e : expr) :
  Pe e = true -> terms_inv (State e [] [] empty_static_env empty_heap no_reads).
Proof.
  intros He. unfold terms_inv. simpl_state.
  refine (conj He (conj I _)). intros id c [=].
Qed.

Lemma terms_inv_run (n : nat) :
  ∀ s s' : state, terms_inv s -> run n s = Step s' -> terms_inv s'.
Proof.
  induction n as [ | n IH]; intros s s' Hinv; simpl.
  - intros [= <-]. exact Hinv.
  - destruct (step s) as [next | k vs | ] eqn:Hstep; try discriminate.
    exact (IH next s' (terms_inv_step s next Hinv Hstep)).
Qed.

End Terms_kept.

(* ======================================================================== *)
(* The binders of the code that a well-formed unit binds                    *)
(* ======================================================================== *)

Theorem well_formed_code_binders : well_formed_code_binders_statement.
Proof.
  intros u e n s id c Hwf Hrun Hlookup.
  pose proof (well_formed_unit_binders_distinct u e Hwf) as Hbinders.
  assert (Hinv :
    terms_inv binders_distinct handler_binders_distinct
      code_binders_distinct s).
  { refine
      (terms_inv_run binders_distinct handler_binders_distinct
         code_binders_distinct _ _ _ _ _ _ _ n _ s
         (terms_inv_initial _ _ _ e Hbinders) Hrun).
    - intros x kind d body H. exact H.
    - intros bindings body H. simpl in H.
      apply andb_true_iff in H as [Hbindings Hbody].
      split; [ exact Hbody | ]. intros id' c' Hin.
      rewrite forallb_forall in Hbindings. exact (Hbindings _ Hin).
    - intros k h body H. simpl in H. apply andb_true_iff in H. exact H.
    - intros handlers body H. simpl in H.
      apply andb_true_iff in H as [H Hbody].
      apply andb_true_iff in H as [_ Hhandlers]. split; assumption.
    - intros ac. reflexivity.
    - intros params metadata handler H. simpl in H.
      apply andb_true_iff in H as [_ H]. exact H.
    - intros [params my_closure ret exn metadata body] H. simpl in H |- *.
      apply andb_true_iff in H as [_ H]. exact H. }
  destruct Hinv as (_ & _ & Hcode). exact (Hcode id c Hlookup).
Qed.

Theorem well_formed_call_binds_my_closure :
  well_formed_call_binds_my_closure_statement.
Proof.
  intros u e n s s' ap callee called slot_values c Hwf Hrun Hexpr Hcallee
    Hclosure Hcode Hcount Hstep.
  pose proof (well_formed_code_binders u e n s called c Hwf Hrun Hcode)
    as Hc.
  rewrite (state_eta s), Hexpr, step_apply, Hcallee in Hstep.
  destruct (eval_simples (state_vars s) (ap_args ap)) as [vs | ];
    [ | discriminate ].
  destruct (eval_simples (state_vars s) (exn_extra_simples ap))
    as [extra | ];
    [ | discriminate ].
  apply
    (call_binds_my_closure callee _ vs _ _ _ _ _ _ rd_ s' called slot_values c
       Hstep Hclosure Hcode);
    [ | exact Hcount ].
  destruct c as [params my_closure ret exn metadata body].
  simpl in Hc |- *.
  apply andb_true_iff in Hc as [Hc _]. apply andb_true_iff in Hc as [Hc _].
  apply andb_true_iff in Hc as [Hc _]. apply negb_true_iff in Hc. exact Hc.
Qed.

(* ======================================================================== *)
(* The set that a piece of code belongs to                                  *)
(* ======================================================================== *)

(* What is found in a map is among what the map gives. *)
Lemma lookup_among_snd (name : positive) (c : positive)
    (bindings : list (positive * positive)) :
  lookup name bindings = Some c -> among c (map snd bindings) = true.
Proof.
  intros Hlookup. unfold among. apply existsb_exists. exists c.
  split; [ | apply Pos.eqb_refl ].
  change c with (snd (name, c)). apply in_map, lookup_in, Hlookup.
Qed.

Section Sets_are.

Variable c : code_id.
Variable function_decls : list (function_slot * code_id).
Variable closure_symbols : option (list (function_slot * symbol)).

Let Pv (v : value) : Prop :=
  value_sets_are c function_decls closure_symbols v = true.
Let Qv (allocated : statically_allocated_value) : Prop :=
  allocated_sets_are c function_decls closure_symbols allocated = true.

(* A set that has the code and passes the test is the one set. *)
Lemma set_is_decls (decls : list (function_slot * code_id))
    (symbols : option (list (function_slot * symbol))) :
  set_is c function_decls closure_symbols decls symbols = true ->
  among c (map snd decls) = true ->
  decls = function_decls
  ∧ ∀ expected : list (function_slot * symbol),
      closure_symbols = Some expected -> symbols = Some expected.
Proof.
  unfold set_is. intros H Hamong. rewrite Hamong in H. simpl in H.
  apply andb_true_iff in H as [Hdecls Hsymbols].
  destruct (function_decls_eq_dec decls function_decls) as [-> | ];
    [ | discriminate ].
  split; [ reflexivity | ].
  intros expected ->. destruct symbols as [given | ]; [ | discriminate ].
  destruct (closure_symbols_eq_dec given expected) as [-> | ];
    [ reflexivity | discriminate ].
Qed.

(* The terms. *)
Lemma sets_are_terms_inv_step (s s' : state) :
  terms_inv (sets_are c function_decls closure_symbols)
    (handler_sets_are c function_decls closure_symbols)
    (code_sets_are c function_decls closure_symbols) s ->
  step s = Step s' ->
  terms_inv (sets_are c function_decls closure_symbols)
    (handler_sets_are c function_decls closure_symbols)
    (code_sets_are c function_decls closure_symbols) s'.
Proof.
  apply terms_inv_step.
  - intros x kind d body H. simpl in H. apply andb_true_iff in H as [_ H].
    exact H.
  - intros bindings body H. simpl in H.
    apply andb_true_iff in H as [Hbindings Hbody].
    split; [ exact Hbody | ]. intros id code Hin.
    rewrite forallb_forall in Hbindings. exact (Hbindings _ Hin).
  - intros k h body H. simpl in H. apply andb_true_iff in H. exact H.
  - intros handlers body H. simpl in H. apply andb_true_iff in H. exact H.
  - intros ac. reflexivity.
  - intros params metadata handler H. exact H.
  - intros [params my_closure ret exn metadata body] H. exact H.
Qed.

(* The values. *)
Lemma slot_values_sets_are_in (slot_values : list (value_slot * value)) :
  forallb
    (fun '(_, contents) =>
       value_sets_are c function_decls closure_symbols contents)
    slot_values
  = true ->
  ∀ (w : value_slot) (contents : value),
    (w, contents) ∈ slot_values -> Pv contents.
Proof.
  intros H w contents Hin. rewrite forallb_forall in H.
  exact (H (w, contents) Hin).
Qed.

Lemma statics_sets_are_lookup (statics : static_env) (s : symbol)
    (allocated : statically_allocated_value) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  lookup s (static_values statics) = Some allocated ->
  Qv allocated.
Proof.
  intros Hstatics Hs. rewrite Forall_forall in Hstatics.
  exact (Hstatics (s, allocated) (lookup_in _ _ _ Hs)).
Qed.

Lemma eval_simple_sets_are (vars : env) (s : simple) (v : value) :
  env_values_all Pv vars -> eval_simple vars s = Some v -> Pv v.
Proof.
  intros Hvars. destruct s as [x | sym | k]; simpl.
  - exact (env_values_all_lookup _ vars x v Hvars).
  - intros [= <-]. reflexivity.
  - intros [= <-]. reflexivity.
Qed.

Lemma eval_simples_sets_are (vars : env) :
  env_values_all Pv vars ->
  ∀ (args : list simple) (vs : list value),
    eval_simples vars args = Some vs ->
    forallb (value_sets_are c function_decls closure_symbols) vs = true.
Proof.
  intros Hvars. induction args as [ | s args IH]; intros vs; simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_sets_are vars s v Hvars Hs). exact (IH rest eq_refl).
Qed.

Lemma eval_value_slots_sets_are (vars : env) :
  env_values_all Pv vars ->
  ∀ (value_slots : list (value_slot * simple))
    (slot_values : list (value_slot * value)),
    eval_value_slots vars value_slots = Some slot_values ->
    forallb
      (fun '(_, contents) =>
         value_sets_are c function_decls closure_symbols contents)
      slot_values
    = true.
Proof.
  intros Hvars.
  induction value_slots as [ | [slot s] value_slots IH]; intros slot_values;
    simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_value_slots vars value_slots) as [rest | ];
      [ | discriminate ].
    simpl. intros [= <-]. simpl.
    rewrite (eval_simple_sets_are vars s v Hvars Hs). exact (IH rest eq_refl).
Qed.

Lemma values_sets_are_Forall (vs : list value) :
  forallb (value_sets_are c function_decls closure_symbols) vs = true ->
  Forall Pv vs.
Proof.
  intros H. rewrite forallb_forall in H. apply Forall_forall. exact H.
Qed.

(* A closure that passes the test, read as a closure of a set: if its
   set has the code it is the one set, static if that is asked; and the
   values of its value slots pass the test. *)
Lemma closure_in_set_sets_are (statics : static_env) (v : value)
    (slot : function_slot) (decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  Pv v ->
  closure_in_set statics v = Some (slot, decls, slot_values) ->
  (among c (map snd decls) = true ->
   decls = function_decls
   ∧ ∀ expected : list (function_slot * symbol),
       closure_symbols = Some expected ->
       ∃ s : symbol,
         v = Value_symbol s
         ∧ lookup s (static_values statics)
           = Some (Static_closure_of_set slot expected decls slot_values))
  ∧ ∀ (w : value_slot) (contents : value),
      (w, contents) ∈ slot_values -> Pv contents.
Proof.
  intros Hstatics Hv. unfold closure_in_set.
  destruct v as [k | code slots | at_slot set_decls slots | s | tag fields
    | elements | contents | l | pa_callee pa_args pa_arity];
    [ | | | | discriminate .. ].
  - discriminate.
  - unfold Pv in Hv. simpl in Hv. apply andb_true_iff in Hv as [Hset Hslots].
    intros [= <- <- <-].
    split; [ | exact (slot_values_sets_are_in slots Hslots) ].
    intros Hamong. destruct (set_is_decls _ _ Hset Hamong) as [Hdecls Hnone].
    split; [ exact Hdecls | ]. intros expected Hexpected.
    discriminate (Hnone expected Hexpected).
  - unfold Pv in Hv. simpl in Hv. apply andb_true_iff in Hv as [Hset Hslots].
    destruct (lookup at_slot set_decls); [ | discriminate ].
    simpl. intros [= <- <- <-].
    split; [ | exact (slot_values_sets_are_in slots Hslots) ].
    intros Hamong. destruct (set_is_decls _ _ Hset Hamong) as [Hdecls Hnone].
    split; [ exact Hdecls | ]. intros expected Hexpected.
    discriminate (Hnone expected Hexpected).
  - destruct (lookup s (static_values statics))
      as [[code slots | at_slot syms set_decls slots | tag fields | contents
          | bytes | elements | ] | ] eqn:Hs;
      try discriminate.
    + pose proof (statics_sets_are_lookup statics s _ Hstatics Hs) as Hvalid.
      unfold Qv in Hvalid. simpl in Hvalid.
      apply andb_true_iff in Hvalid as [Hset Hslots].
      intros [= <- <- <-].
      split; [ | exact (slot_values_sets_are_in slots Hslots) ].
      intros Hamong.
      destruct (set_is_decls _ _ Hset Hamong) as [Hdecls Hnone].
      split; [ exact Hdecls | ]. intros expected Hexpected.
      discriminate (Hnone expected Hexpected).
    + pose proof (statics_sets_are_lookup statics s _ Hstatics Hs) as Hvalid.
      unfold Qv in Hvalid. simpl in Hvalid.
      apply andb_true_iff in Hvalid as [Hset Hslots].
      destruct (lookup at_slot set_decls); [ | discriminate ].
      simpl. intros [= <- <- <-].
      split; [ | exact (slot_values_sets_are_in slots Hslots) ].
      intros Hamong.
      destruct (set_is_decls _ _ Hset Hamong) as [Hdecls Hsymbols].
      split; [ exact Hdecls | ]. intros expected Hexpected.
      injection (Hsymbols expected Hexpected) as ->.
      exists s. split; [ reflexivity | exact Hs ].
Qed.

Lemma value_sets_are_of_numbers : holds_of_numbers Pv.
Proof. intros k. split; reflexivity. Qed.

Lemma value_sets_are_of_blocks : holds_of_blocks Pv.
Proof.
  intros tag fields.
  exact (forallb_forall (value_sets_are c function_decls closure_symbols)
           fields).
Qed.

Lemma value_sets_are_of_arrays : holds_of_arrays Pv.
Proof.
  intros elements.
  exact (forallb_forall (value_sets_are c function_decls closure_symbols)
           elements).
Qed.

Lemma value_sets_are_in_static_blocks (statics : static_env) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  holds_in_static_blocks statics Pv.
Proof.
  intros Hstatics s tag fields _ Hs v Hin.
  pose proof (statics_sets_are_lookup statics s _ Hstatics Hs) as Hvalid.
  unfold Qv in Hvalid. simpl in Hvalid. rewrite forallb_forall in Hvalid.
  exact (Hvalid v Hin).
Qed.

Lemma value_sets_are_in_static_arrays (statics : static_env) :
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  holds_in_static_arrays statics Pv.
Proof.
  intros Hstatics s elements _ Hs v Hin.
  pose proof (statics_sets_are_lookup statics s _ Hstatics Hs) as Hvalid.
  unfold Qv in Hvalid. simpl in Hvalid. rewrite forallb_forall in Hvalid.
  exact (Hvalid v Hin).
Qed.

Lemma project_function_slot_sets_are (statics : static_env)
    (move_from move_to : function_slot) (v w : value) :
  Pv v -> project_function_slot statics move_from move_to v = Some w -> Pv w.
Proof.
  intros Hv. unfold project_function_slot.
  destruct v as [k | code slots | at_slot decls slots | s | tag fields
    | elements | contents | l | pa_callee pa_args pa_arity];
    [ | | | | discriminate .. ].
  - discriminate.
  - intros Hproject.
    apply project_unnamed_function_slot in Hproject as (_ & _ & ->).
    exact Hv.
  - destruct (Pos.eqb at_slot move_from); [ | discriminate ].
    destruct (lookup at_slot decls); [ | discriminate ].
    destruct (lookup move_to decls); [ | discriminate ].
    simpl. intros [= <-]. exact Hv.
  - destruct (lookup s (static_values statics))
      as [[code slots | at_slot syms decls slots | tag fields | contents
          | bytes | elements | ] | ];
      try discriminate.
    + intros Hproject.
      apply project_unnamed_function_slot in Hproject as (_ & _ & ->).
      reflexivity.
    + destruct (Pos.eqb at_slot move_from); [ | discriminate ].
      destruct (lookup at_slot decls); [ | discriminate ].
      destruct (lookup move_to decls); [ | discriminate ].
      destruct (lookup move_to syms); [ | discriminate ].
      simpl. intros [= <-]. reflexivity.
Qed.

Lemma eval_named_sets_are (statics : static_env) (vars : env) (d : named)
    (v : value) :
  named_set_is c function_decls closure_symbols d = true ->
  env_values_all Pv vars ->
  Forall (fun binding => Qv (snd binding)) (static_values statics) ->
  eval_named statics vars d = Some v ->
  Pv v.
Proof.
  intros Hd Hvars Hstatics.
  destruct d as [s | p args | code slots | slot decls slots | slot closure
    | project_from slot closure | move_from move_to closure];
    simpl.
  - exact (eval_simple_sets_are vars s v Hvars).
  - destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
      [ | discriminate ].
    simpl. intros Hprim.
    apply (eval_prim_preserves Pv statics p vs v value_sets_are_of_numbers
             value_sets_are_of_blocks value_sets_are_of_arrays
             (value_sets_are_in_static_blocks statics Hstatics)
             (value_sets_are_in_static_arrays statics Hstatics));
      [ | exact Hprim ].
    pose proof (eval_simples_sets_are vars Hvars args vs Hargs) as Hvs.
    rewrite forallb_forall in Hvs. exact Hvs.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Pv. simpl. simpl in Hd. rewrite Hd.
    exact (eval_value_slots_sets_are vars Hvars slots vs Hslots).
  - destruct (lookup slot decls); [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Pv. simpl. simpl in Hd. rewrite Hd.
    exact (eval_value_slots_sets_are vars Hvars slots vs Hslots).
  - destruct (eval_simple vars closure) as [k | ] eqn:Hk; [ | discriminate ].
    simpl.
    destruct (closure_of statics k) as [[code slot_values] | ] eqn:Hclosure;
      [ | discriminate ].
    simpl. intros Hlookup.
    destruct (proj1 (closure_in_set_of statics k code slot_values) Hclosure)
      as (at_slot & decls & Hin_set & _).
    destruct (closure_in_set_sets_are statics k at_slot decls slot_values
                Hstatics (eval_simple_sets_are vars closure k Hvars Hk)
                Hin_set)
      as [_ Hslots].
    exact (Hslots slot v (lookup_in _ _ _ Hlookup)).
  - destruct (eval_simple vars closure) as [k | ] eqn:Hk; [ | discriminate ].
    simpl. unfold project_value_slot.
    destruct (closure_in_set statics k) as [[[at_slot decls] slot_values] | ]
      eqn:Hin_set;
      [ | discriminate ].
    simpl. destruct (Pos.eqb at_slot project_from); [ | discriminate ].
    intros Hlookup.
    destruct (closure_in_set_sets_are statics k at_slot decls slot_values
                Hstatics (eval_simple_sets_are vars closure k Hvars Hk)
                Hin_set)
      as [_ Hslots].
    exact (Hslots slot v (lookup_in _ _ _ Hlookup)).
  - destruct (eval_simple vars closure) as [k | ] eqn:Hk; [ | discriminate ].
    simpl.
    exact (project_function_slot_sets_are statics move_from move_to k v
             (eval_simple_sets_are vars closure k Hvars Hk)).
Qed.

Lemma eval_static_const_sets_are (const : static_const) (vars : env)
    (allocated : statically_allocated_value) :
  static_const_set_is c function_decls closure_symbols const = true ->
  env_values_all Pv vars ->
  eval_static_const vars const = Some allocated ->
  Qv allocated.
Proof.
  intros Hconst Hvars.
  destruct const as [code slots | slot syms decls slots | tag fields
    | contents | bytes | elements];
    simpl.
  - destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Qv. simpl. simpl in Hconst. rewrite Hconst.
    exact (eval_value_slots_sets_are vars Hvars slots vs Hslots).
  - destruct (lookup slot decls); [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ] eqn:Hslots;
      [ | discriminate ].
    simpl. intros [= <-]. unfold Qv. simpl. simpl in Hconst. rewrite Hconst.
    exact (eval_value_slots_sets_are vars Hvars slots vs Hslots).
  - destruct (eval_simples vars fields) as [vs | ] eqn:Hfields;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simples_sets_are vars Hvars fields vs Hfields).
  - destruct (eval_simple vars contents) as [v | ] eqn:Hcontents;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simple_sets_are vars contents v Hvars Hcontents).
  - intros [= <-]. reflexivity.
  - destruct (eval_simples vars elements) as [vs | ] eqn:Helements;
      [ | discriminate ].
    simpl. intros [= <-].
    exact (eval_simples_sets_are vars Hvars elements vs Helements).
Qed.

Lemma eval_static_bindings_sets_are (vars : env) :
  env_values_all Pv vars ->
  ∀ (bindings : list static_binding) (statics statics' : static_env),
    forallb (binding_sets_are c function_decls closure_symbols) bindings
    = true ->
    Forall (fun binding => Qv (snd binding)) (static_values statics) ->
    eval_static_bindings vars bindings statics = Some statics' ->
    Forall (fun binding => Qv (snd binding)) (static_values statics').
Proof.
  intros Hvars.
  induction bindings as [ | [id code | id | sym const] bindings IH];
    intros statics statics'; simpl.
  - intros _ Hvalid [= <-]. exact Hvalid.
  - intros H Hvalid. apply andb_true_iff in H as [_ Hrest].
    destruct (lookup id (static_code statics)); [ discriminate | ].
    apply (IH _ _ Hrest). exact Hvalid.
  - intros Hrest Hvalid. exact (IH _ _ Hrest Hvalid).
  - intros H Hvalid. apply andb_true_iff in H as [Hconst Hrest].
    destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ] eqn:Heval;
      simpl; [ | discriminate ].
    apply (IH _ _ Hrest). simpl. apply Forall_cons; [ | exact Hvalid ].
    exact (eval_static_const_sets_are const vars allocated Hconst Hvars Heval).
Qed.

Lemma value_sets_are_partial (callee : value) (args : list value)
    (args_arity : arity) :
  Pv (Value_partial_application callee args args_arity)
  <-> Pv callee ∧ Forall Pv args.
Proof.
  unfold Pv. simpl. rewrite andb_true_iff, forallb_Forall. reflexivity.
Qed.

Lemma state_sets_are_step (s : state) :
  sets_are c function_decls closure_symbols (state_expr s) = true ->
  state_sets_are c function_decls closure_symbols s ->
  match step s with
  | Step s' => state_sets_are c function_decls closure_symbols s'
  | Exit _ vs _ _ => Forall Pv vs
  | Undefined => True
  end.
Proof.
  intros P Hvalid.
  apply (step_values_all
           (fun _ v => value_sets_are c function_decls closure_symbols v = true)
           (fun _ allocated =>
              allocated_sets_are c function_decls closure_symbols allocated
              = true)
           s (fun _ => value_sets_are_partial) Hvalid).
  - intros size tuple fields Htuple Hfields.
    destruct Hvalid as (_ & _ & Hstatics & _).
    apply (tuple_fields_forall _ (state_statics s) size tuple fields);
      [ | exact Hfields ].
    intros field v Hprim.
    refine (eval_prim_preserves Pv (state_statics s) _ [tuple] v
              value_sets_are_of_numbers value_sets_are_of_blocks
              value_sets_are_of_arrays
              (value_sets_are_in_static_blocks (state_statics s) Hstatics)
              (value_sets_are_in_static_arrays (state_statics s) Hstatics) _
              Hprim).
    intros w [<- | []]. exact Htuple.
  - intros x kind d body v hp' rd' Hexpr Hexec. rewrite Hexpr in P. simpl in P.
    apply andb_true_iff in P as [Pd _].
    destruct Hvalid as (Hvars & _ & Hstatics & Hheap).
    refine
      (exec_named_values_all Pv (state_statics s) (state_heap s) hp'
         (state_reads s) rd'
              (state_vars s) d v value_sets_are_of_numbers
              value_sets_are_of_blocks value_sets_are_of_arrays
              (value_sets_are_in_static_blocks (state_statics s) Hstatics)
              (value_sets_are_in_static_arrays (state_statics s) Hstatics)
              (fun _ => eq_refl) Hheap _ _ Hexec).
    + intros p args vs _ Hvs.
      exact (proj1 (forallb_forall _ _)
               (eval_simples_sets_are (state_vars s) Hvars args vs Hvs)).
    + intros _ w Heval.
      exact (eval_named_sets_are (state_statics s) (state_vars s) d w Pd
               Hvars Hstatics Heval).
  - intros bindings body statics' Hexpr Heval.
    split; [ exact (fun v H => H) | ].
    rewrite Hexpr in P. simpl in P. apply andb_true_iff in P as [Pbindings _].
    destruct Hvalid as (Hvars & _ & Hstatics & _).
    exact (eval_static_bindings_sets_are (state_vars s) Hvars bindings
             (state_statics s) statics' Pbindings Hstatics Heval).
  - intros ap callee vs extra _ Hcallee Hargs Hextra.
    destruct Hvalid as (Hvars & _).
    split; [ | split ].
    + exact (eval_simple_sets_are (state_vars s) (ap_callee ap) callee Hvars
               Hcallee).
    + exact (values_sets_are_Forall vs
               (eval_simples_sets_are (state_vars s) Hvars (ap_args ap) vs
                  Hargs)).
    + exact (values_sets_are_Forall extra
               (eval_simples_sets_are (state_vars s) Hvars
                  (exn_extra_simples ap) extra Hextra)).
  - intros ac vs _ Hargs. destruct Hvalid as (Hvars & _).
    exact (values_sets_are_Forall vs
             (eval_simples_sets_are (state_vars s) Hvars (ac_args ac) vs
                Hargs)).
Qed.

Lemma state_sets_are_run (n : nat) :
  ∀ s : state,
    terms_inv (sets_are c function_decls closure_symbols)
      (handler_sets_are c function_decls closure_symbols)
      (code_sets_are c function_decls closure_symbols) s ->
    state_sets_are c function_decls closure_symbols s ->
    match run n s with
    | Step s' => state_sets_are c function_decls closure_symbols s'
    | Exit _ vs _ _ => Forall Pv vs
    | Undefined => True
    end.
Proof.
  induction n as [ | n IH]; intros s Hterms Hvalid; simpl.
  - exact Hvalid.
  - pose proof (state_sets_are_step s (proj1 Hterms) Hvalid) as Hstep.
    destruct (step s) as [next | k vs | ] eqn:Hnext.
    + exact (IH next (sets_are_terms_inv_step s next Hterms Hnext) Hstep).
    + exact Hstep.
    + exact I.
Qed.

End Sets_are.

Theorem sets_are_kept : sets_are_kept_statement.
Proof.
  intros c function_decls closure_symbols e n s He Hrun.
  pose proof
    (state_sets_are_run c function_decls closure_symbols n
       (State e [] [] empty_static_env empty_heap no_reads)
       (terms_inv_initial _ _ _ e He)
       (conj (Forall_nil _)
          (conj I (conj (Forall_nil _) (heap_values_all_empty _)))))
    as H.
  rewrite Hrun in H. exact H.
Qed.

Theorem closure_of_code : closure_of_code_statement.
Proof.
  intros c function_decls closure_symbols s closure v slot_values
    (Hvars & _ & Hstatics & _) Hv Hclosure.
  destruct (proj1 (closure_in_set_of (state_statics s) v c slot_values)
              Hclosure)
    as (slot & decls & Hin_set & Hslot).
  destruct (closure_in_set_sets_are c function_decls closure_symbols
              (state_statics s) v slot decls slot_values Hstatics
              (eval_simple_sets_are c function_decls closure_symbols
                 (state_vars s) closure v Hvars Hv)
              Hin_set)
    as [Hdecls _].
  destruct (Hdecls (lookup_among_snd slot c decls Hslot)) as [-> _].
  exists slot. split; [ exact Hin_set | exact Hslot ].
Qed.

(* In the run of a well-formed unit all of whose sets with the code are
   the one set, a closure of the code is at the slot that the set gives
   the code. *)
Lemma closure_of_code_at_slot (u : unit_interface) (c : code_id)
    (function_decls : list (function_slot * code_id))
    (closure_symbols : option (list (function_slot * symbol))) (e : expr)
    (n : nat) (s : state) (closure : simple) (v : value)
    (slot_values : list (value_slot * value)) (slot : function_slot) :
  well_formed_unit u e = true ->
  sets_are c function_decls closure_symbols e = true ->
  run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
  eval_simple (state_vars s) closure = Some v ->
  closure_of (state_statics s) v = Some (c, slot_values) ->
  lookup slot function_decls = Some c ->
  closure_in_set (state_statics s) v
  = Some (slot, function_decls, slot_values).
Proof.
  intros Hwf He Hrun Hv Hclosure Hslot.
  pose proof (sets_are_kept c function_decls closure_symbols e n s He Hrun)
    as Hsets.
  destruct (closure_of_code c function_decls closure_symbols s closure v
              slot_values Hsets Hv Hclosure)
    as (at_slot & Hin_set & _).
  destruct (well_formed_values_valid u e n s Hwf Hrun)
    as (Hvars & _ & Hstatics & _).
  destruct (closure_in_set_valid (state_statics s) v at_slot function_decls
              slot_values Hstatics
              (eval_simple_valid (state_vars s) closure v Hvars Hv) Hin_set)
    as [Hset _].
  unfold set_well_formed in Hset. apply andb_true_iff in Hset as [_ Hcodes].
  rewrite (closure_code_determines_slot (state_statics s) v at_slot slot
             function_decls slot_values slot_values c Hin_set Hcodes Hclosure
             Hslot).
  exact Hin_set.
Qed.

Theorem sibling_of_code : sibling_of_code_statement.
Proof.
  intros u c function_decls closure_symbols e n s closure v
    slot_values slot sibling sibling_code Hwf He Hrun Hv Hclosure Hslot
    Hsibling.
  pose proof
    (closure_of_code_at_slot u c function_decls closure_symbols e n s
       closure v slot_values slot Hwf He Hrun Hv Hclosure Hslot)
    as Hin_set.
  destruct (well_formed_values_valid u e n s Hwf Hrun)
    as (_ & _ & Hstatics & _).
  assert (Hbound : is_bound sibling function_decls = true)
    by (unfold is_bound; rewrite Hsibling; reflexivity).
  destruct (project_function_slot_valid (state_statics s) slot sibling v
              function_decls slot_values Hstatics Hin_set Hbound)
    as [w Hproject].
  exists w. split; [ exact Hproject | ].
  destruct (project_function_slot_bound (state_statics s) slot sibling v w
              (well_formed_sets_bound u e n s Hwf Hrun) Hproject)
    as (decls' & slots' & Hin_set' & Hw).
  rewrite Hin_set in Hin_set'. injection Hin_set' as <- <-.
  apply (proj2 (closure_in_set_of (state_statics s) w sibling_code
                  slot_values)).
  exists sibling, function_decls. split; [ exact Hw | exact Hsibling ].
Qed.

Theorem sibling_symbol_of_code : sibling_symbol_of_code_statement.
Proof.
  intros u c function_decls symbols e n s closure v slot_values slot
    sibling sibling_code Hwf He Hrun Hv Hclosure Hslot Hsibling.
  pose proof
    (closure_of_code_at_slot u c function_decls (Some symbols) e n s
       closure v slot_values slot Hwf He Hrun Hv Hclosure Hslot)
    as Hin_set.
  destruct (sets_are_kept c function_decls (Some symbols) e n s He Hrun)
    as (Hsets_vars & _ & Hsets_statics & _).
  destruct (closure_in_set_sets_are c function_decls (Some symbols)
              (state_statics s) v slot function_decls slot_values
              Hsets_statics
              (eval_simple_sets_are c function_decls (Some symbols)
                 (state_vars s) closure v Hsets_vars Hv)
              Hin_set)
    as [Hone _].
  destruct (Hone (lookup_among_snd slot c function_decls Hslot))
    as [_ Hstatic].
  destruct (Hstatic symbols eq_refl) as (slot_symbol & -> & Hbound_symbol).
  destruct (well_formed_sets_bound u e n s Hwf Hrun slot_symbol slot
              symbols function_decls slot_values Hbound_symbol)
    as [Hslot_symbol _].
  destruct (well_formed_values_valid u e n s Hwf Hrun)
    as (_ & _ & Hstatics & _).
  pose proof
    (statics_valid_lookup (state_statics s) slot_symbol _ Hstatics
       Hbound_symbol)
    as Hvalid.
  simpl in Hvalid. apply andb_true_iff in Hvalid as [Hvalid _].
  apply andb_true_iff in Hvalid as [_ Hnames].
  destruct (list_eq_dec Pos.eq_dec (map fst symbols) (map fst function_decls))
    as [Hsame | ]; [ | discriminate ].
  assert (Hsibling_bound : is_bound sibling symbols = true).
  { apply (is_bound_same_names sibling symbols function_decls Hsame).
    unfold is_bound. rewrite Hsibling. reflexivity. }
  unfold is_bound in Hsibling_bound.
  destruct (lookup sibling symbols) as [sibling_symbol | ]
    eqn:Hsibling_symbol; [ | discriminate ].
  exists slot_symbol, sibling_symbol.
  split; [ exact Hslot_symbol | ]. split; [ reflexivity | ].
  split; [ reflexivity | ].
  unfold project_function_slot. cbv beta iota zeta.
  rewrite Hbound_symbol, Pos.eqb_refl, Hslot, Hsibling. simpl.
  rewrite Hsibling_symbol. reflexivity.
Qed.

Theorem set_of_code_bound : set_of_code_bound_statement.
Proof.
  intros u c function_decls symbols e n s closure v slot_values Hwf
    He Hrun Hv Hclosure.
  pose proof (sets_are_kept c function_decls (Some symbols) e n s He Hrun)
    as Hsets.
  destruct (closure_of_code c function_decls (Some symbols) s closure v
              slot_values Hsets Hv Hclosure)
    as (slot & Hin_set & Hslot).
  destruct Hsets as (Hsets_vars & _ & Hsets_statics & _).
  destruct (closure_in_set_sets_are c function_decls (Some symbols)
              (state_statics s) v slot function_decls slot_values
              Hsets_statics
              (eval_simple_sets_are c function_decls (Some symbols)
                 (state_vars s) closure v Hsets_vars Hv)
              Hin_set)
    as [Hone _].
  destruct (Hone (lookup_among_snd slot c function_decls Hslot))
    as [_ Hstatic].
  destruct (Hstatic symbols eq_refl) as (slot_symbol & -> & Hbound_symbol).
  exact (proj2 (well_formed_sets_bound u e n s Hwf Hrun slot_symbol
                  slot symbols function_decls slot_values Hbound_symbol)).
Qed.

(* ======================================================================== *)
(* The example                                                              *)
(* ======================================================================== *)

(* The two units are run. *)
Theorem parity_let : parity_let_statement.
Proof.
  intros i answer [[= <- <-] | [[= <- <-] | []]]; eexists;
    (split; [ reflexivity | ]); exists 60; eexists; vm_compute; reflexivity.
Qed.

Theorem parity_static : parity_static_statement.
Proof.
  intros i answer [[= <- <-] | [[= <- <-] | []]]; exists 60; eexists;
    vm_compute;
    reflexivity.
Qed.

(* The conditions that do not look at the numbers are computed; the
   argument is the one number of the two units that is not given. *)
Theorem parity_well_formed : parity_well_formed_statement.
Proof.
  intros i Hi. split.
  - apply well_formed_unit_intro;
      [ vm_compute; reflexivity ..
      | idtac
      | vm_compute; reflexivity
      | vm_compute; reflexivity ].
    unfold FlambdaClosuresSpec.parity_static. cbn [numbers_in_range].
    apply andb_true_iff. split; [ vm_compute; reflexivity | ].
    unfold apply_in_range. cbn. rewrite Hi. reflexivity.
  - eexists. split; [ reflexivity | ].
    apply well_formed_unit_intro;
      [ vm_compute; reflexivity ..
      | idtac
      | vm_compute; reflexivity
      | vm_compute; reflexivity ].
    cbn [numbers_in_range].
    apply andb_true_iff. split; [ vm_compute; reflexivity | ].
    apply andb_true_iff. split; [ vm_compute; reflexivity | ].
    apply andb_true_iff. split; [ vm_compute; reflexivity | ].
    unfold apply_in_range. cbn. rewrite Hi. reflexivity.
Qed.

Theorem parity_sets_are : parity_sets_are_statement.
Proof.
  intros i c [<- | [<- | []]];
    (split; [ vm_compute; reflexivity | ]);
    eexists; (split; [ reflexivity | vm_compute; reflexivity ]).
Qed.

Theorem partial_static_set : partial_static_set_statement.
Proof.
  split;
    [ exists 10; eexists; vm_compute; reflexivity | vm_compute; reflexivity ].
Qed.

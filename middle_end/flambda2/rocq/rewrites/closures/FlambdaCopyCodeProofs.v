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

(* Proofs of the statements of FlambdaCopyCodeSpec.v. Nothing here
   needs review.

   The unit and the unit with the copies are run side by side, in
   step. The second state is the first with its terms rewritten, the
   code that its static environment binds bound a second time under
   the new ids, and the closures that the symbols given are bound to
   given the copies ([copy_state]); the values of its variables are
   those of the first. A code id that is bound in the first run is
   bound to the rewritten code in the second, and its copy to the
   same: no new id is bound in the first run, which a run keeps
   ([copied_inv]). A direct call of the first run that is defined is
   of a closure of the code named; in the second run that closure
   has the code or its copy, and the call names what the closure has
   or is indirect. No primitive looks at the code of a closure
   (FlambdaPrimitiveProofs.eval_prim_related, with equality for the
   relation). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaNominalSpec FlambdaClosuresProofs
  FlambdaWellFormedSpec FlambdaUnusedStaticsSpec FlambdaImportsSpec
  FlambdaStaticsInScopeSpec FlambdaObserversSpec FlambdaUnitFromProofs
  FlambdaEquivarianceTactic FlambdaCopyCodeSpec.
Import ListNotations.

(* ======================================================================== *)
(* Lists and maps                                                           *)
(* ======================================================================== *)

Lemma lookup_map_snd {A B : Type} (f : A -> B) (name : positive)
    (l : list (positive * A)) :
  lookup name (map (fun binding => (fst binding, f (snd binding))) l)
  = option_map f (lookup name l).
Proof.
  induction l as [ | [k a] l IH]; simpl; [ reflexivity | ].
  destruct (Pos.eqb name k); [ reflexivity | exact IH ].
Qed.

Lemma lookup_map_keyed {A B : Type} (f : positive -> A -> B)
    (name : positive) (l : list (positive * A)) :
  lookup name
    (map (fun binding => (fst binding, f (fst binding) (snd binding))) l)
  = option_map (f name) (lookup name l).
Proof.
  induction l as [ | [k a] l IH]; simpl; [ reflexivity | ].
  destruct (Pos.eqb_spec name k) as [-> | Hne]; [ reflexivity | exact IH ].
Qed.

Lemma forall2_eq_refl {A : Type} (l : list A) : Forall2 eq l l.
Proof. induction l; constructor; [ reflexivity | assumption ]. Qed.

Lemma forall2_eq {A : Type} (l l' : list A) : Forall2 eq l l' -> l = l'.
Proof. induction 1; [ reflexivity | congruence ]. Qed.

(* What an association list gives is among its values. *)
Lemma lookup_among_snd (name value : code_id)
    (l : list (code_id * code_id)) :
  lookup name l = Some value -> among value (map snd l) = true.
Proof.
  induction l as [ | [k v] l IH]; [ discriminate | ].
  cbn [map snd]. rewrite lookup_cons, among_cons.
  destruct (Pos.eqb name k).
  - intros [= ->]. rewrite Pos.eqb_refl. reflexivity.
  - intros H. rewrite (IH H). apply orb_true_r.
Qed.

(* With distinct values, a value is given to one name only. *)
Lemma lookup_injective (a b value : code_id)
    (l : list (code_id * code_id)) :
  distinct (map snd l) = true ->
  lookup a l = Some value -> lookup b l = Some value -> a = b.
Proof.
  induction l as [ | [k v] l IH]; [ discriminate | ].
  cbn [map snd distinct]. rewrite !lookup_cons.
  intros Hdistinct. apply andb_true_iff in Hdistinct as [Hfresh Hdistinct].
  apply negb_true_iff in Hfresh.
  destruct (Pos.eqb_spec a k) as [-> | Ha];
    destruct (Pos.eqb_spec b k) as [-> | Hb].
  - reflexivity.
  - intros [= ->] H. rewrite (lookup_among_snd b value l H) in Hfresh.
    discriminate.
  - intros H [= ->]. rewrite (lookup_among_snd a value l H) in Hfresh.
    discriminate.
  - exact (IH Hdistinct).
Qed.

Section Copied.

Variable ids : list (code_id * code_id).
Variable symbols : list symbol.

Hypothesis Hnew : distinct (map snd ids) = true.


(* ======================================================================== *)
(* The second state                                                         *)
(* ======================================================================== *)

Definition copy_allocated (s : symbol)
    (allocated : statically_allocated_value) : statically_allocated_value :=
  if among s symbols
  then
    match allocated with
    | Static_closure code slots => Static_closure (copy_of ids code) slots
    | Static_closure_of_set slot closure_symbols decls slots =>
      Static_closure_of_set slot closure_symbols (copy_decls ids decls) slots
    | Static_block _ _ | Static_boxed_number _ | Static_string _
    | Static_immutable_array _ | Static_imported _ =>
      allocated
    end
  else allocated.

(* The code bound under an id, rewritten, and under the copy of the id
   if it has one: the copy is bound last, as [copy_binding] binds it. *)
Definition copy_code_binding (binding : code_id * code)
    : list (code_id * code) :=
  match lookup (fst binding) ids with
  | Some id' =>
    [(id', copy_code ids symbols (snd binding));
     (fst binding, copy_code ids symbols (snd binding))]
  | None => [(fst binding, copy_code ids symbols (snd binding))]
  end.

Definition copy_statics (statics : static_env) : static_env :=
  Static_env (flat_map copy_code_binding (static_code statics))
    (map
       (fun binding =>
          (fst binding, copy_allocated (fst binding) (snd binding)))
       (static_values statics)).

Fixpoint copy_cont (c : cont_value) : cont_value :=
  match c with
  | Closure h vars conts recursive =>
    Closure (copy_handler ids symbols h) vars
      (map (fun binding => (fst binding, copy_cont (snd binding))) conts)
      (map (fun '(k, h) => (k, copy_handler ids symbols h)) recursive)
  | Outer k => Outer k
  | Extra_args c extra => Extra_args (copy_cont c) extra
  | Over_application remaining remaining_arity return_arity return_to exn_to =>
    Over_application remaining remaining_arity return_arity
      (copy_cont return_to) (copy_cont exn_to)
  end.

Definition copy_conts (conts : cont_env) : cont_env :=
  map (fun binding => (fst binding, copy_cont (snd binding))) conts.

Definition copy_state (s : state) : state :=
  State (copy_expr ids symbols (state_expr s)) (state_vars s)
    (copy_conts (state_conts s)) (copy_statics (state_statics s))
      (state_heap s) (state_reads s).

(* No new id is bound. *)
Definition no_new (codes : list (code_id * code)) : Prop :=
  ∀ id : code_id, among id (map snd ids) = true -> lookup id codes = None.

Lemma no_new_cons (id : code_id) (c : code) (codes : list (code_id * code)) :
  no_new ((id, c) :: codes) -> among id (map snd ids) = false ∧ no_new codes.
Proof.
  intros H. split.
  - destruct (among id (map snd ids)) eqn:Hid; [ | reflexivity ].
    specialize (H id Hid). rewrite lookup_head in H. discriminate.
  - intros id' Hid'. specialize (H id' Hid'). rewrite lookup_cons in H.
    destruct (Pos.eqb id' id); [ discriminate | exact H ].
Qed.

Lemma lookup_copy_values (s : symbol) (statics : static_env) :
  lookup s (static_values (copy_statics statics))
  = option_map (copy_allocated s) (lookup s (static_values statics)).
Proof. unfold copy_statics. apply lookup_map_keyed. Qed.

(* A code id that is not new is bound to the rewritten code. *)
Lemma lookup_copy_code (id : code_id) (codes : list (code_id * code)) :
  no_new codes ->
  among id (map snd ids) = false ->
  lookup id (flat_map copy_code_binding codes)
  = option_map (copy_code ids symbols) (lookup id codes).
Proof.
  intros Hcodes Hid.
  induction codes as [ | [k c] codes IH]; [ reflexivity | ].
  destruct (no_new_cons k c codes Hcodes) as [Hk Hrest].
  cbn [flat_map]. unfold copy_code_binding at 1. cbn [fst snd].
  rewrite lookup_cons.
  destruct (lookup k ids) as [k' | ] eqn:Hcopy; cbn [app].
  - rewrite !lookup_cons.
    destruct (Pos.eqb_spec id k') as [-> | Hne'].
    + rewrite (lookup_among_snd k k' ids Hcopy) in Hid. discriminate.
    + destruct (Pos.eqb id k); [ reflexivity | exact (IH Hrest) ].
  - rewrite lookup_cons.
    destruct (Pos.eqb id k); [ reflexivity | exact (IH Hrest) ].
Qed.

(* The copy of a code id is bound to the rewritten code of the id. *)
Lemma lookup_copy_code_new (id id' : code_id)
    (codes : list (code_id * code)) :
  no_new codes ->
  lookup id ids = Some id' ->
  lookup id' (flat_map copy_code_binding codes)
  = option_map (copy_code ids symbols) (lookup id codes).
Proof.
  intros Hcodes Hid.
  pose proof (lookup_among_snd id id' ids Hid) as Hid'.
  induction codes as [ | [k c] codes IH]; [ reflexivity | ].
  destruct (no_new_cons k c codes Hcodes) as [Hk Hrest].
  cbn [flat_map]. unfold copy_code_binding at 1. cbn [fst snd].
  rewrite lookup_cons.
  assert (Hne : Pos.eqb id' k = false).
  { destruct (Pos.eqb_spec id' k) as [-> | ]; [ | reflexivity ].
    rewrite Hid' in Hk. discriminate. }
  destruct (lookup k ids) as [k' | ] eqn:Hcopy; cbn [app].
  - rewrite !lookup_cons, Hne.
    destruct (Pos.eqb_spec id' k') as [-> | Hne'].
    + rewrite (lookup_injective id k k' ids Hnew Hid Hcopy), Pos.eqb_refl.
      reflexivity.
    + destruct (Pos.eqb_spec id k) as [-> | ]; [ | exact (IH Hrest) ].
      rewrite Hid in Hcopy. injection Hcopy as ->. contradiction.
  - rewrite lookup_cons, Hne.
    destruct (Pos.eqb_spec id k) as [-> | ]; [ | exact (IH Hrest) ].
    rewrite Hid in Hcopy. discriminate.
Qed.

(* ======================================================================== *)
(* The observations                                                         *)
(* ======================================================================== *)

Lemma lookup_copy_decls (slot : function_slot)
    (decls : list (function_slot * code_id)) :
  lookup slot (copy_decls ids decls)
  = option_map (copy_of ids) (lookup slot decls).
Proof. unfold copy_decls. apply lookup_map_snd. Qed.

(* What a symbol is bound to in the second static environment, whether
   or not the environment has been unfolded. *)
Ltac lookup_copied :=
  first [ rewrite lookup_copy_values | rewrite lookup_map_keyed ].

(* What is not a closure is what it was. *)
Ltac same_observation statics s :=
  lookup_copied;
  destruct (lookup s (static_values statics)) as [[] | ]; simpl;
  try reflexivity; unfold copy_allocated;
  destruct (among s symbols); reflexivity.

Lemma block_of_copy (statics : static_env) (v : value) :
  block_of (copy_statics statics) v = block_of statics v.
Proof. destruct v; simpl; try reflexivity. same_observation statics s. Qed.

Lemma boxed_number_of_copy (statics : static_env) (v : value) :
  boxed_number_of (copy_statics statics) v = boxed_number_of statics v.
Proof. destruct v; simpl; try reflexivity. same_observation statics s. Qed.

Lemma string_of_copy (statics : static_env) (v : value) :
  string_of (copy_statics statics) v = string_of statics v.
Proof. destruct v; simpl; try reflexivity. same_observation statics s. Qed.

Lemma array_of_copy (statics : static_env) (v : value) :
  array_of (copy_statics statics) v = array_of statics v.
Proof. destruct v; simpl; try reflexivity. same_observation statics s. Qed.

Lemma bound_if_symbol_copy (statics : static_env) (v : value) :
  bound_if_symbol (copy_statics statics) v = bound_if_symbol statics v.
Proof.
  destruct v; simpl; try reflexivity. lookup_copied.
  destruct (lookup s (static_values statics)); reflexivity.
Qed.

Lemma imported_of_copy (statics : static_env) (v : value) :
  imported_of (copy_statics statics) v = imported_of statics v.
Proof. destruct v; simpl; try reflexivity. same_observation statics s. Qed.

Lemma project_function_slot_copy (statics : static_env)
    (move_from move_to : function_slot) (v : value) :
  project_function_slot (copy_statics statics) move_from move_to v
  = project_function_slot statics move_from move_to v.
Proof.
  destruct v; simpl; try reflexivity. lookup_copied.
  destruct (lookup s (static_values statics))
    as [[ acode aslots | aslot asyms adecls aslots | | | | | ] | ];
    simpl; try reflexivity; unfold copy_allocated;
    destruct (among s symbols); try reflexivity.
  destruct (Pos.eqb aslot move_from); [ | reflexivity ].
  rewrite !lookup_copy_decls.
  destruct (lookup aslot adecls); [ | reflexivity ].
  destruct (lookup move_to adecls); reflexivity.
Qed.

(* The code of a closure in the second run: the copy of its code for
   one of the symbols, its code otherwise. *)
Definition code_at (v : value) (code : code_id) : code_id :=
  match v with
  | Value_symbol s => if among s symbols then copy_of ids code else code
  | _ => code
  end.

Definition decls_at (v : value) (decls : list (function_slot * code_id))
    : list (function_slot * code_id) :=
  match v with
  | Value_symbol s => if among s symbols then copy_decls ids decls else decls
  | _ => decls
  end.

Lemma closure_of_copy (statics : static_env) (v : value) (code : code_id)
    (slots : list (value_slot * value)) :
  closure_of statics v = Some (code, slots) ->
  closure_of (copy_statics statics) v = Some (code_at v code, slots).
Proof.
  destruct v
    as [ c | vcode vslots | vslot vdecls vslots | s | tag fields | elements | c
       | loc | callee args args_arity ];
    simpl; try discriminate; try exact (fun H => H).
  lookup_copied.
  destruct (lookup s (static_values statics))
    as [[ acode aslots | aslot asyms adecls aslots | | | | | ] | ];
    simpl; try discriminate; unfold copy_allocated;
    destruct (among s symbols); simpl; try exact (fun H => H).
  - intros [= <- <-]. reflexivity.
  - rewrite lookup_copy_decls.
    destruct (lookup aslot adecls); simpl; [ | discriminate ].
    intros [= <- <-]. reflexivity.
Qed.

Lemma closure_in_set_copy (statics : static_env) (v : value)
    (at_slot : function_slot) (decls : list (function_slot * code_id))
    (slots : list (value_slot * value)) :
  closure_in_set statics v = Some (at_slot, decls, slots) ->
  ∃ decls' : list (function_slot * code_id),
    closure_in_set (copy_statics statics) v = Some (at_slot, decls', slots).
Proof.
  destruct v
    as [ c | vcode vslots | vslot vdecls vslots | s | tag fields | elements | c
       | loc | callee args args_arity ];
    simpl; try discriminate; try (intros H; eexists; exact H).
  lookup_copied.
  destruct (lookup s (static_values statics))
    as [[ acode aslots | aslot asyms adecls aslots | | | | | ] | ];
    simpl; try discriminate; unfold copy_allocated;
    destruct (among s symbols); simpl; try (intros H; eexists; exact H).
  - intros [= <- <- <-]. eexists. reflexivity.
  - rewrite lookup_copy_decls.
    destruct (lookup aslot adecls); simpl; [ | discriminate ].
    intros [= <- <- <-]. eexists. reflexivity.
Qed.

(* No observation that a primitive makes tells the two static
   environments apart, so no primitive does. *)
Lemma copy_statics_respects (statics : static_env) :
  respects_observations eq statics (copy_statics statics).
Proof.
  split; [ | split ].
  - split; intros; subst; reflexivity.
  - split; [ intros c; split; reflexivity | split ].
    + intros tag fields fields' Hfields.
      rewrite (forall2_eq fields fields' Hfields). reflexivity.
    + intros elements elements' Helements.
      rewrite (forall2_eq elements elements' Helements). reflexivity.
  - intros v v' <-.
    split;
      [ | split; [ | split; [ | split; [ | split; [ | split; [ | split ] ] ] ] ]
      ].
    + intros tag fields H. exists fields. rewrite block_of_copy.
      split; [ exact H | apply forall2_eq_refl ].
    + intros c. rewrite boxed_number_of_copy. exact (fun H => H).
    + intros bytes. rewrite string_of_copy. exact (fun H => H).
    + destruct (closure_of statics v) as [[code slots] | ] eqn:Hc;
        [ | congruence ].
      rewrite (closure_of_copy statics v code slots Hc).
      intros _. discriminate.
    + exact (fun H => H).
    + rewrite bound_if_symbol_copy. exact (fun H => H).
    + intros s. rewrite imported_of_copy. exact (fun H => H).
    + intros elements H. exists elements. rewrite array_of_copy.
      split; [ exact H | apply forall2_eq_refl ].
Qed.

Lemma eval_prim_copy (statics : static_env) (p : prim) (vs : list value)
    (v : value) :
  eval_prim statics p vs = Some v ->
  eval_prim (copy_statics statics) p vs = Some v.
Proof.
  intros Hp.
  destruct
    (eval_prim_related eq statics (copy_statics statics) p vs vs v
       (copy_statics_respects statics) (forall2_eq_refl vs) Hp)
    as (v' & Hp' & <-).
  exact Hp'.
Qed.

(* ======================================================================== *)
(* Evaluation                                                               *)
(* ======================================================================== *)

Lemma eval_named_copy (statics : static_env) (vars : env) (d : named)
    (v : value) :
  eval_named statics vars d = Some v ->
  eval_named (copy_statics statics) vars d = Some v.
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure]; simpl;
    try exact (fun H => H).
  - destruct (eval_simples vars args) as [vs | ]; simpl; [ | discriminate ].
    apply eval_prim_copy.
  - destruct (eval_simple vars closure) as [w | ]; simpl; [ | discriminate ].
    destruct (closure_of statics w) as [[called cslots] | ] eqn:Hc; simpl;
      [ | discriminate ].
    rewrite (closure_of_copy statics w called cslots Hc). simpl.
    exact (fun H => H).
  - destruct (eval_simple vars closure) as [w | ]; simpl; [ | discriminate ].
    unfold project_value_slot.
    destruct (closure_in_set statics w) as [[[at_slot decls] cslots] | ]
      eqn:Hc; simpl; [ | discriminate ].
    destruct (closure_in_set_copy statics w at_slot decls cslots Hc)
      as (decls' & ->).
    simpl. exact (fun H => H).
  - destruct (eval_simple vars closure) as [w | ]; simpl; [ | discriminate ].
    rewrite project_function_slot_copy. exact (fun H => H).
Qed.

(* The same where the named expression is run on a heap: what is not a
   primitive has its value by [eval_named], and no primitive tells the
   two static environments apart on a heap either
   (FlambdaHeapSpec.exec_prim_related_statement, at equality). *)
Lemma exec_named_copy (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (d : named) (v : value) :
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  exec_named (copy_statics statics) hp rd vars d = Some (v, hp', rd').
Proof.
  intros Hd.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try (pose proof (fun H => exec_named_pure _ _ _ _ _ _ _ _ H Hd) as Hpure;
         destruct (Hpure eq_refl) as (Hw & -> & ->);
         rewrite exec_named_not_prim by (intros ? ?; discriminate);
         rewrite (eval_named_copy statics vars _ v Hw); reflexivity).
  apply exec_named_prim_some in Hd as (vs & Hvs & Hexec & ->).
  apply exec_named_prim_some. exists vs.
  split; [ exact Hvs | split; [ | reflexivity ] ].
  assert (Hlocations : FlambdaHeapSpec.relates_locations eq).
  { split; [ intros l; reflexivity | ].
    split; [ intros l w H; symmetry; exact H | intros w l H; exact H ]. }
  destruct
    (FlambdaHeapProofs.exec_prim_related eq statics (copy_statics statics) hp
       hp hp' p vs vs v (copy_statics_respects statics) Hlocations
       (proj2 (FlambdaHeapProofs.heap_alike_eq hp hp) eq_refl)
       (forall2_eq_refl vs) Hexec)
    as (v' & hp1 & Hexec' & <- & Halike).
  rewrite (proj1 (FlambdaHeapProofs.heap_alike_eq hp' hp1) Halike).
  exact Hexec'.
Qed.

Lemma eval_static_const_copy (vars : env) (s : symbol) (const : static_const)
    (allocated : statically_allocated_value) :
  eval_static_const vars const = Some allocated ->
  eval_static_const vars
    (if among s symbols then copy_static_const ids const else const)
  = Some (copy_allocated s allocated).
Proof.
  unfold copy_allocated.
  destruct (among s symbols);
    [ | intros H; rewrite H; destruct allocated; reflexivity ].
  destruct const
    as [code slots | fslot syms decls slots | tag fields | contents | bytes
        | elements];
    simpl.
  - destruct (eval_value_slots vars slots) as [vs | ]; simpl;
      [ | discriminate ].
    intros [= <-]. reflexivity.
  - rewrite lookup_copy_decls.
    destruct (lookup fslot decls); simpl; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ]; simpl;
      [ | discriminate ].
    intros [= <-]. reflexivity.
  - destruct (eval_simples vars fields); simpl; [ | discriminate ].
    intros [= <-]. reflexivity.
  - destruct (eval_simple vars contents); simpl; [ | discriminate ].
    intros [= <-]. reflexivity.
  - intros [= <-]. reflexivity.
  - destruct (eval_simples vars elements); simpl; [ | discriminate ].
    intros [= <-]. reflexivity.
Qed.

(* The static environment with the code of the copies. *)
Lemma copy_statics_code (statics : static_env) :
  static_code (copy_statics statics)
  = flat_map copy_code_binding (static_code statics).
Proof. reflexivity. Qed.

(* A group of bindings that binds no new id, from a static environment
   that binds none: the rewritten group gives the second environment,
   and the environment that results binds no new id. *)
Lemma eval_static_bindings_copy (vars : env)
    (bindings : list static_binding) :
  ∀ statics statics' : static_env,
    forallb (binding_binds_no_code (map snd ids)) bindings = true ->
    no_new (static_code statics) ->
    eval_static_bindings vars bindings statics = Some statics' ->
    eval_static_bindings vars (flat_map (copy_binding ids symbols) bindings)
      (copy_statics statics)
    = Some (copy_statics statics')
    ∧ no_new (static_code statics').
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics statics' Hbinds Hno; cbn [flat_map].
  - simpl. intros [= <-]. split; [ reflexivity | exact Hno ].
  - cbn [forallb binding_binds_no_code] in Hbinds.
    apply andb_true_iff in Hbinds as [Hc Hbinds].
    apply andb_true_iff in Hc as [Hid _]. apply negb_true_iff in Hid.
    cbn [eval_static_bindings].
    destruct (lookup id (static_code statics)) eqn:Hbound; [ discriminate | ].
    intros Heval.
    assert (Hno' : no_new ((id, c) :: static_code statics)).
    { intros id' Hid'. rewrite lookup_cons.
      destruct (Pos.eqb_spec id' id) as [-> | ]; [ | exact (Hno id' Hid') ].
      rewrite Hid in Hid'. discriminate. }
    destruct
      (IH (Static_env ((id, c) :: static_code statics)
             (static_values statics))
          statics' Hbinds Hno' Heval)
      as [Hcopied Hno''].
    split; [ | exact Hno'' ].
    cbn [copy_binding]. rewrite <- Hcopied.
    unfold copy_statics at 2. cbn [static_code static_values flat_map].
    unfold copy_code_binding at 1. cbn [fst snd].
    destruct (lookup id ids) as [id' | ] eqn:Hcopy; cbn [app];
      cbn [eval_static_bindings];
      rewrite copy_statics_code, (lookup_copy_code id _ Hno Hid), Hbound;
      cbn [option_map static_code static_values]; [ | reflexivity ].
    rewrite lookup_cons.
    pose proof (lookup_among_snd id id' ids Hcopy) as Hid'.
    destruct (Pos.eqb_spec id' id) as [-> | ];
      [ rewrite Hid in Hid'; discriminate | ].
    rewrite (lookup_copy_code_new id id' _ Hno Hcopy), Hbound.
    reflexivity.
  - cbn [forallb] in Hbinds. apply andb_true_iff in Hbinds as [_ Hbinds].
    cbn [copy_binding app eval_static_bindings]. apply IH; assumption.
  - cbn [forallb] in Hbinds. apply andb_true_iff in Hbinds as [_ Hbinds].
    cbn [copy_binding app eval_static_bindings].
    lookup_copied.
    destruct (lookup sym (static_values statics)); simpl; [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ] eqn:Hconst;
      simpl; [ | discriminate ].
    rewrite (eval_static_const_copy vars sym const allocated Hconst). simpl.
    intros Heval.
    exact (IH (Static_env (static_code statics)
                 ((sym, allocated) :: static_values statics))
              statics' Hbinds Hno Heval).
Qed.

(* ======================================================================== *)
(* Terms and continuations                                                  *)
(* ======================================================================== *)

(* The rewrite keeps whether a term binds something static. *)
Lemma copy_binding_binds (binding : static_binding) :
  existsb binding_binds (copy_binding ids symbols binding)
  = binding_binds binding.
Proof.
  destruct binding as [id c | id | s const]; cbn [copy_binding];
    [ | reflexivity | reflexivity ].
  destruct (lookup id ids); reflexivity.
Qed.

Fixpoint copy_expr_binds_statics (e : expr) {struct e} :
  binds_statics (copy_expr ids symbols e) = binds_statics e
with copy_handler_binds_statics (h : continuation_handler) {struct h} :
  handler_binds_statics (copy_handler ids symbols h)
  = handler_binds_statics h.
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply copy_expr_binds_statics.
    + rewrite copy_expr_binds_statics. f_equal.
      induction bindings as [ | binding bindings IH]; [ reflexivity | ].
      cbn [flat_map existsb].
      rewrite existsb_app, copy_binding_binds, IH. reflexivity.
    + rewrite copy_handler_binds_statics, copy_expr_binds_statics.
      reflexivity.
    + rewrite copy_expr_binds_statics. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite copy_handler_binds_statics, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply copy_expr_binds_statics.
Qed.

Lemma copy_handlers_bind_statics
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, copy_handler ids symbols h)) handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite copy_handler_binds_statics, IH. reflexivity.
Qed.

Lemma resolve_copy (k : continuation) (conts : cont_env) :
  resolve k (copy_conts conts) = copy_cont (resolve k conts).
Proof.
  unfold resolve, copy_conts. rewrite lookup_map_snd.
  destruct (lookup k conts); reflexivity.
Qed.

Lemma rec_closures_copy
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env) :
  rec_closures (map (fun '(k, h) => (k, copy_handler ids symbols h)) handlers)
    vars (copy_conts conts)
  = copy_conts (rec_closures handlers vars conts).
Proof.
  unfold rec_closures, copy_conts. rewrite !map_map.
  apply map_ext. intros [k h]. reflexivity.
Qed.

Lemma with_extra_args_copy (c : cont_value) (extra : list value) :
  copy_cont (with_extra_args c extra) = with_extra_args (copy_cont c) extra.
Proof. destruct extra; reflexivity. Qed.

Lemma copy_code_parts (c : code) :
  code_params (copy_code ids symbols c) = code_params c
  ∧ code_my_closure (copy_code ids symbols c) = code_my_closure c
  ∧ code_return_continuation (copy_code ids symbols c)
    = code_return_continuation c
  ∧ code_exn_continuation (copy_code ids symbols c) = code_exn_continuation c
  ∧ code_params_arity (copy_code ids symbols c) = code_params_arity c
  ∧ code_result_arity (copy_code ids symbols c) = code_result_arity c
  ∧ code_body (copy_code ids symbols c) = copy_expr ids symbols (code_body c).
Proof. destruct c. simpl. auto 8. Qed.

(* ======================================================================== *)
(* Calls                                                                    *)
(* ======================================================================== *)

(* What the second run does where the first does something: the same
   step, to the second state of the result, or the same exit. *)
Definition result_copied (result result' : step_result) : Prop :=
  match result with
  | Step t => result' = Step (copy_state t)
  | Exit k vs hp rd => result' = Exit k vs hp rd
  | Undefined => True
  end.

(* A call of the second run has the shape of the call of the first,
   with the same values, of the code rewritten. *)
Definition copy_shape (shape : call_shape) : call_shape :=
  match shape with
  | Full_call closure c vs args_arity =>
    Full_call closure (copy_code ids symbols c) vs args_arity
  | Over_call closure c vs remaining args_arity remaining_arity =>
    Over_call closure (copy_code ids symbols c) vs remaining args_arity
      remaining_arity
  | Partial_call partial => Partial_call partial
  end.

(* Whether a call of a kind is one of a closure of the code given. *)
Definition expected (kind : call_kind) (called : code_id) : bool :=
  match kind with
  | Direct named => Pos.eqb called named
  | Indirect => true
  end.

(* The call kind of the second run answers that of the first, for a
   callee: an indirect call stays one, and a call that is of the code
   of the callee is of the code that the callee has in the second
   run. *)
Definition kind_follows (statics : static_env) (callee : value)
    (kind kind' : call_kind) : Prop :=
  (kind = Indirect -> kind' = Indirect)
  ∧ (∀ (called : code_id) (slots : list (value_slot * value)),
      closure_of statics callee = Some (called, slots) ->
      expected kind called = true ->
      expected kind' (code_at callee called) = true).

Lemma kind_follows_indirect (statics : static_env) (callee : value) :
  kind_follows statics callee Indirect Indirect.
Proof. split; [ exact (fun H => H) | reflexivity ]. Qed.

(* The code of a closure of the first run is bound, in the second, to
   the rewritten code, under the code that the closure has there. *)
Lemma lookup_code_at (v : value) (called : code_id) (c : code)
    (codes : list (code_id * code)) :
  no_new codes ->
  lookup called codes = Some c ->
  lookup (code_at v called) (flat_map copy_code_binding codes)
  = Some (copy_code ids symbols c).
Proof.
  intros Hno Hc.
  assert (Hold : among called (map snd ids) = false).
  { destruct (among called (map snd ids)) eqn:H; [ | reflexivity ].
    rewrite (Hno called H) in Hc. discriminate. }
  assert (Hsame :
    lookup called (flat_map copy_code_binding codes)
    = Some (copy_code ids symbols c)).
  { rewrite (lookup_copy_code called codes Hno Hold), Hc. reflexivity. }
  destruct v; simpl; try exact Hsame.
  destruct (among s symbols); [ | exact Hsame ].
  unfold copy_of. destruct (lookup called ids) as [id' | ] eqn:Hcopy;
    [ | exact Hsame ].
  rewrite (lookup_copy_code_new called id' codes Hno Hcopy), Hc. reflexivity.
Qed.

Lemma shape_of_closure_call_copy (statics : static_env) (closure : value)
    (kind kind' : call_kind) (vs : list value)
    (args_arity return_arity : arity) (shape : call_shape) :
  no_new (static_code statics) ->
  kind_follows statics closure kind kind' ->
  shape_of_closure_call statics closure kind vs args_arity return_arity
  = Some shape ->
  shape_of_closure_call (copy_statics statics) closure kind' vs args_arity
    return_arity
  = Some (copy_shape shape).
Proof.
  intros Hno [_ Hkind]. unfold shape_of_closure_call.
  destruct (closure_of statics closure) as [[called slots] | ] eqn:Hc;
    [ | discriminate ].
  rewrite (closure_of_copy statics closure called slots Hc).
  specialize (Hkind called slots eq_refl). unfold expected in Hkind.
  destruct
    (match kind with
     | Direct named => Pos.eqb called named
     | Indirect => true
     end);
    [ | discriminate ].
  rewrite (Hkind eq_refl).
  destruct (lookup called (static_code statics)) as [c | ] eqn:Hcode;
    [ | discriminate ].
  rewrite copy_statics_code,
    (lookup_code_at closure called c (static_code statics) Hno Hcode).
  cbv iota zeta.
  destruct (copy_code_parts c)
    as (_ & _ & _ & _ & Hparams_arity & Hresult_arity & _).
  assert (Htupled :
    code_is_tupled (copy_code ids symbols c) = code_is_tupled c)
    by (destruct c; reflexivity).
  destruct (arguments_for_code statics c vs args_arity)
    as [[vs' args_arity'] | ] eqn:Hargs; [ | discriminate ].
  rewrite (arguments_for_code_mono statics (copy_statics statics) c
             (copy_code ids symbols c) vs args_arity _ Htupled Hparams_arity
             (fun tuple rest p v _ => eval_prim_copy statics p [tuple] v)
             Hargs).
  simpl. unfold shape_of_code_call. cbv zeta.
  rewrite Hparams_arity, Hresult_arity, Htupled.
  destruct (Nat.eqb (length args_arity') (length (code_params_arity c)));
    [ intros [= <-]; reflexivity | ].
  destruct
    (negb
       (kinds_eqb (map kind_of_value vs') (unarize args_arity')
        && partial_or_over_application args_arity' return_arity
             (code_params_arity c) (code_result_arity c)));
    [ discriminate | ].
  destruct (Nat.ltb (length (code_params_arity c)) (length args_arity'));
    [ intros [= <-]; reflexivity | ].
  destruct (code_is_tupled c); [ discriminate | ].
  intros [= <-]. reflexivity.
Qed.

Lemma shape_of_call_copy (statics : static_env) (callee : value)
    (kind kind' : call_kind) (vs : list value)
    (args_arity return_arity : arity) (shape : call_shape) :
  no_new (static_code statics) ->
  kind_follows statics callee kind kind' ->
  shape_of_call statics callee kind vs args_arity return_arity = Some shape ->
  shape_of_call (copy_statics statics) callee kind' vs args_arity return_arity
  = Some (copy_shape shape).
Proof.
  intros Hno Hkind. unfold shape_of_call.
  destruct kind as [named | ].
  - (* A direct call: its callee is a closure, and no partial
       application. *)
    intros Hshape.
    assert (Hpartial : partial_application_of callee = None).
    { unfold shape_of_closure_call in Hshape.
      destruct callee; try reflexivity. discriminate Hshape. }
    rewrite Hpartial.
    assert (Hgoal :
      shape_of_closure_call (copy_statics statics) callee kind' vs
        args_arity return_arity
      = Some (copy_shape shape))
      by exact (shape_of_closure_call_copy statics callee (Direct named)
                  kind' vs args_arity return_arity shape Hno Hkind Hshape).
    destruct kind'; exact Hgoal.
  - rewrite (proj1 Hkind eq_refl).
    destruct (partial_application_of callee)
      as [[[closure applied] applied_arity] | ];
      [ | exact (shape_of_closure_call_copy statics callee Indirect Indirect
                   vs args_arity return_arity shape Hno
                   (kind_follows_indirect statics callee)) ].
    destruct args_arity as [ | component args_arity]; [ discriminate | ].
    destruct (kinds_eqb _ _); [ | discriminate ].
    exact (shape_of_closure_call_copy statics closure Indirect Indirect _ _
             return_arity shape Hno (kind_follows_indirect statics closure)).
Qed.

(* A full call. *)
Lemma call_code_copy (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  result_copied
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure (copy_code ids symbols c) vs args_arity return_arity
       (copy_cont return_to) (copy_cont exn_to) (copy_statics statics) hp rd).
Proof.
  unfold result_copied, call_code.
  destruct (copy_code_parts c)
    as (Hparams & Hmy_closure & Hret & Hexn & Hparams_arity & Hresult_arity
        & Hbody).
  rewrite Hparams, Hmy_closure, Hret, Hexn, Hbody, copy_expr_binds_statics.
  destruct (binds_statics (code_body c)); [ exact I | ].
  rewrite Hparams_arity, Hresult_arity.
  match goal with
  | |- context [negb ?agrees] => destruct (negb agrees)
  end;
    [ exact I | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)]);
    [ | exact I ].
  reflexivity.
Qed.

(* A call of a given shape, given what the two runs do with a partial
   application. *)
Lemma run_call_copy (return_partial return_partial' : value -> step_result)
    (shape : call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  (∀ partial : value,
    result_copied (return_partial partial) (return_partial' partial)) ->
  result_copied
    (run_call return_partial (Some shape) return_arity return_to exn_to
       statics hp rd)
    (run_call return_partial' (Some (copy_shape shape)) return_arity
       (copy_cont return_to) (copy_cont exn_to) (copy_statics statics) hp rd).
Proof.
  intros Hpartial.
  destruct shape
    as [closure c vs args_arity
       | closure c vs remaining args_arity remaining_arity | partial];
    simpl.
  - apply call_code_copy.
  - destruct (copy_code_parts c) as (_ & _ & _ & _ & _ & -> & _).
    exact (call_code_copy closure c vs args_arity (code_result_arity c)
             (Over_application remaining remaining_arity return_arity
                return_to exn_to)
             exn_to statics hp rd).
  - apply Hpartial.
Qed.

(* A call, given what the two runs do when the return continuation is
   applied to a partial application. *)
Lemma call_function_copy_from (callee : value) (kind kind' : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  no_new (static_code statics) ->
  kind_follows statics callee kind kind' ->
  (∀ partial : value,
    result_copied (apply_cont_value return_to [partial] statics hp rd)
      (apply_cont_value (copy_cont return_to) [partial]
         (copy_statics statics) hp rd)) ->
  result_copied
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee kind' vs args_arity return_arity
       (copy_cont return_to) (copy_cont exn_to) (copy_statics statics) hp rd).
Proof.
  intros Hno Hkind Hpartial. unfold call_function.
  destruct (shape_of_call statics callee kind vs args_arity return_arity)
    as [shape | ] eqn:Hshape;
    [ | exact I ].
  rewrite (shape_of_call_copy statics callee kind kind' vs args_arity
             return_arity shape Hno Hkind Hshape).
  apply run_call_copy. exact Hpartial.
Qed.

Lemma apply_cont_value_copy (c : cont_value) :
  ∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record),
    no_new (static_code statics) ->
    result_copied (apply_cont_value c vs statics hp rd)
      (apply_cont_value (copy_cont c) vs (copy_statics statics) hp rd).
Proof.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IH extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros vs statics hp rd Hno.
  - unfold result_copied. simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs def_vars) as [handler_vars | ];
      [ | exact I ].
    unfold copy_state. simpl_state. f_equal. f_equal.
    fold (copy_conts def_conts). rewrite rec_closures_copy.
    unfold copy_conts. rewrite map_app. reflexivity.
  - reflexivity.
  - apply IH. exact Hno.
  - simpl copy_cont. rewrite !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ exact I | | exact I ].
    apply call_function_copy_from;
      [ exact Hno | apply kind_follows_indirect | ].
    intros partial. apply IHreturn. exact Hno.
Qed.

Lemma call_function_copy (callee : value) (kind kind' : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  no_new (static_code statics) ->
  kind_follows statics callee kind kind' ->
  result_copied
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee kind' vs args_arity return_arity
       (copy_cont return_to) (copy_cont exn_to) (copy_statics statics) hp rd).
Proof.
  intros Hno Hkind.
  apply call_function_copy_from; [ exact Hno | exact Hkind | ].
  intros partial. apply apply_cont_value_copy. exact Hno.
Qed.

(* ======================================================================== *)
(* A step                                                                   *)
(* ======================================================================== *)

(* The rewrite of a call changes its call kind and nothing else. *)
Lemma copy_call_parts (ap : apply_expr) :
  ap_callee (copy_call ids symbols ap) = ap_callee ap
  ∧ ap_args (copy_call ids symbols ap) = ap_args ap
  ∧ ap_continuation (copy_call ids symbols ap) = ap_continuation ap
  ∧ ap_exn_continuation (copy_call ids symbols ap) = ap_exn_continuation ap
  ∧ exn_extra_simples (copy_call ids symbols ap) = exn_extra_simples ap
  ∧ ap_args_arity (copy_call ids symbols ap) = ap_args_arity ap
  ∧ ap_return_arity (copy_call ids symbols ap) = ap_return_arity ap.
Proof.
  destruct ap as [callee kind args k exn extra args_arity return_arity].
  unfold copy_call. cbn [ap_call_kind ap_callee].
  destruct kind as [id | ]; [ | auto 8 ].
  destruct callee as [x | s | c];
    [ destruct (lookup id ids) | destruct (among s symbols)
    | destruct (lookup id ids) ];
    auto 8.
Qed.

(* A code id that has no copy is the code of a closure in both runs. *)
Lemma code_at_not_copied (v : value) (id : code_id) :
  lookup id ids = None -> code_at v id = id.
Proof.
  intros Hid. destruct v; simpl; try reflexivity.
  destruct (among s symbols); [ | reflexivity ].
  unfold copy_of. rewrite Hid. reflexivity.
Qed.

(* The call kind that the rewrite gives a call answers the kind that
   the call had. *)
Lemma copy_call_kind_follows (statics : static_env) (vars : env)
    (ap : apply_expr) (callee : value) :
  eval_simple vars (ap_callee ap) = Some callee ->
  kind_follows statics callee (ap_call_kind ap)
    (ap_call_kind (copy_call ids symbols ap)).
Proof.
  unfold copy_call.
  destruct (ap_call_kind ap) as [id | ] eqn:Hkind;
    [ | intros _; rewrite Hkind; apply kind_follows_indirect ].
  assert (Hno_symbol :
    ∀ kind' : call_kind,
      kind'
      = match lookup id ids with
        | Some _ => Indirect
        | None => Direct id
        end ->
      kind_follows statics callee (Direct id) kind').
  { intros kind' ->. split; [ discriminate | ].
    intros called slots _ Hexpected.
    destruct (lookup id ids) eqn:Hid; [ reflexivity | ].
    apply Pos.eqb_eq in Hexpected as ->. simpl.
    rewrite (code_at_not_copied callee id Hid). apply Pos.eqb_refl. }
  destruct (ap_callee ap) as [x | s | c].
  - intros _. apply Hno_symbol.
    destruct (lookup id ids); [ reflexivity | exact Hkind ].
  - intros [= <-]. split; [ discriminate | ].
    intros called slots _ Hexpected.
    apply Pos.eqb_eq in Hexpected as ->.
    destruct (among s symbols) eqn:Hs.
    + simpl. rewrite Hs. apply Pos.eqb_refl.
    + rewrite Hkind. simpl. rewrite Hs. apply Pos.eqb_refl.
  - intros _. apply Hno_symbol.
    destruct (lookup id ids); [ reflexivity | exact Hkind ].
Qed.

(* A result of the first run that is answered, where the first run
   keeps its static environment. *)
Lemma result_copied_keeps (result result' : step_result)
    (statics : static_env) :
  no_new (static_code statics) ->
  (∀ t : state, result = Step t -> state_statics t = statics) ->
  result_copied result result' ->
  match result with
  | Step t =>
    result' = Step (copy_state t)
    ∧ no_new (static_code (state_statics t))
  | Exit k vs hp rd => result' = Exit k vs hp rd
  | Undefined => True
  end.
Proof.
  intros Hno Hkeeps Hcopied.
  destruct result as [t | k vs hp rd | ]; simpl in *;
    [ | exact Hcopied | exact I ].
  split; [ exact Hcopied | ]. rewrite (Hkeeps t eq_refl). exact Hno.
Qed.

(* A step of a state whose expression binds no new id, from a static
   environment that binds none, is a step of the second state, to the
   second state of the result; and no new id is bound after it. *)
Lemma copy_step_components (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  binds_no_code (map snd ids) e = true ->
  no_new (static_code statics) ->
  match step (State e vars conts statics hp rd) with
  | Step t =>
    step
      (State (copy_expr ids symbols e) vars (copy_conts conts)
         (copy_statics statics) hp rd)
    = Step (copy_state t)
    ∧ no_new (static_code (state_statics t))
  | Exit k vs hp' rd' =>
    step
      (State (copy_expr ids symbols e) vars (copy_conts conts)
         (copy_statics statics) hp rd)
    = Exit k vs hp' rd'
  | Undefined => True
  end.
Proof.
  intros Hbinds Hno.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl copy_expr; autorewrite with step.
  - (* Let *)
    destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hv;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) xkind) eqn:Hkind; [ | exact I ].
    rewrite (exec_named_copy statics hp hp' rd rd' vars d v Hv), Hkind.
    split; [ reflexivity | exact Hno ].
  - (* Let_static *)
    simpl in Hbinds. apply andb_true_iff in Hbinds as [Hbindings _].
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Heval; [ | exact I ].
    destruct
      (eval_static_bindings_copy vars bindings statics statics' Hbindings Hno
         Heval)
      as [-> Hno'].
    split; [ reflexivity | exact Hno' ].
  - (* Let_cont_non_recursive *)
    split; [ reflexivity | exact Hno ].
  - (* Let_cont_recursive *)
    rewrite copy_handlers_bind_statics.
    destruct (handlers_bind_statics handlers); [ exact I | ].
    split; [ | exact Hno ].
    unfold copy_state. simpl_state. rewrite rec_closures_copy.
    unfold copy_conts. rewrite map_app. reflexivity.
  - (* Apply *)
    destruct (copy_call_parts ap)
      as (-> & -> & -> & -> & -> & -> & ->).
    destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
      [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | exact I ].
    rewrite !resolve_copy, <- with_extra_args_copy.
    apply (result_copied_keeps _ _ statics Hno).
    + intros t Ht.
      exact (call_function_statics _ _ _ _ _ _ _ _ _ rd_ t Ht).
    + apply call_function_copy; [ exact Hno | ].
      exact (copy_call_kind_follows statics vars ap callee Hcallee).
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
    unfold apply_continuation. rewrite resolve_copy.
    apply (result_copied_keeps _ _ statics Hno).
    + intros t Ht. exact (apply_cont_value_statics _ _ _ _ rd_ t Ht).
    + apply apply_cont_value_copy. exact Hno.
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    destruct (find_arm i arms) as [ac | ]; [ | exact I ].
    split; [ reflexivity | exact Hno ].
  - exact I.
Qed.

(* ======================================================================== *)
(* The run of a unit                                                        *)
(* ======================================================================== *)

(* The terms of a state bind no new id, and its static environment
   binds none. *)
Definition copied_inv (s : state) : Prop :=
  terms_inv (binds_no_code (map snd ids))
    (handler_binds_no_code (map snd ids)) (code_binds_no_code (map snd ids)) s
  ∧ no_new (static_code (state_statics s)).

Lemma copy_step (s : state) :
  copied_inv s ->
  match step s with
  | Step t => step (copy_state s) = Step (copy_state t) ∧ copied_inv t
  | Exit k vs hp rd => step (copy_state s) = Exit k vs hp rd
  | Undefined => True
  end.
Proof.
  intros [Hterms Hno].
  pose proof
    (copy_step_components (state_expr s) (state_vars s) (state_conts s)
       (state_statics s) (state_heap s) (state_reads s) (proj1 Hterms) Hno)
    as Hstep.
  rewrite <- (state_eta s) in Hstep.
  assert (Hkept :
    ∀ t : state,
      step s = Step t ->
      terms_inv (binds_no_code (map snd ids))
        (handler_binds_no_code (map snd ids))
        (code_binds_no_code (map snd ids)) t).
  { intros t. revert Hterms. apply terms_inv_step.
    - intros x xkind d body H. exact H.
    - intros bindings body H. simpl in H.
      apply andb_true_iff in H as [Hbindings Hbody].
      split; [ exact Hbody | ]. intros id c Hin.
      pose proof (proj1 (forallb_forall _ _) Hbindings (Bind_code id c) Hin)
        as Hc.
      simpl in Hc. apply andb_true_iff in Hc. tauto.
    - intros k h body H. simpl in H. apply andb_true_iff in H. tauto.
    - intros handlers body H. simpl in H. apply andb_true_iff in H. tauto.
    - intros ac. reflexivity.
    - intros params metadata handler H. exact H.
    - intros [params my_closure ret exn metadata body] H. exact H. }
  destruct (step s) as [t | k vs hp rd | ]; [ | exact Hstep | exact I ].
  destruct Hstep as [Hstep Hno'].
  split; [ exact Hstep | ]. split; [ exact (Hkept t eq_refl) | exact Hno' ].
Qed.

Definition copied_related (s s' : state) : Prop :=
  s' = copy_state s ∧ copied_inv s.

Lemma copied_related_step (s s' : state) :
  copied_related s s' -> result_simulated copied_related (step s) (step s').
Proof.
  intros [-> Hinv].
  pose proof (copy_step s Hinv) as Hstep.
  destruct (step s) as [t | k vs | ]; simpl; [ | | exact I ].
  - destruct Hstep as [-> Hinv']. simpl. split; [ reflexivity | exact Hinv' ].
  - rewrite Hstep. simpl. auto.
Qed.

(* The unit with the copies refines the unit, the two run from a static
   environment that holds no code and binds no closure, and from
   continuations whose terms bind no new id and that the rewrite
   leaves as they are. *)
Lemma copy_expr_refines_from (e : expr) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  binds_no_code (map snd ids) e = true ->
  conts_terms_ok (handler_binds_no_code (map snd ids)) conts ->
  copy_conts conts = conts ->
  static_code statics = [] ->
  copy_statics statics = statics ->
  refines (State e [] conts statics hp rd)
    (State (copy_expr ids symbols e) [] conts statics hp rd).
Proof.
  intros Hbinds Hconts Hsame_conts Hno_code Hsame.
  apply (simulation_refines copied_related copied_related_step).
  split.
  - unfold copy_state. simpl_state. rewrite Hsame, Hsame_conts. reflexivity.
  - split.
    + unfold terms_inv. simpl_state. refine (conj Hbinds (conj Hconts _)).
      intros id c Hlookup. rewrite Hno_code in Hlookup. discriminate Hlookup.
    + simpl_state. intros id _. rewrite Hno_code. reflexivity.
Qed.

(* A term with no static binding in it, which names no symbol and no
   code id, is left as it is, and binds no new id. *)
Fixpoint copy_expr_nothing (e : expr) {struct e} :
  binds_no_static e = true ->
  expr_statics_in_scope [] [] e = true ->
  copy_expr ids symbols e = e.
Proof.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl; try (intros _ _; reflexivity).
  - intros Hno Hscope. apply andb_true_iff in Hscope as [_ Hscope].
    rewrite (copy_expr_nothing body Hno Hscope). reflexivity.
  - discriminate.
  - destruct h as [params metadata handler]. simpl. intros Hno Hscope.
    apply andb_true_iff in Hno as [Hh Hb].
    rewrite !(no_static_names _ body Hb) in Hscope. cbn [app] in Hscope.
    apply andb_true_iff in Hscope as [Hsh Hsb].
    rewrite (copy_expr_nothing handler Hh Hsh),
      (copy_expr_nothing body Hb Hsb).
    reflexivity.
  - intros Hno Hscope. apply andb_true_iff in Hno as [Hhs Hb].
    rewrite !(no_static_names _ body Hb) in Hscope. cbn [app] in Hscope.
    apply andb_true_iff in Hscope as [Hshs Hsb].
    rewrite (copy_expr_nothing body Hb Hsb). f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    simpl in Hhs, Hshs.
    apply andb_true_iff in Hhs as [Hh Hrest].
    apply andb_true_iff in Hshs as [Hsh Hsrest].
    rewrite (copy_expr_nothing handler Hh Hsh), (IH Hrest Hsrest).
    reflexivity.
  - intros _ Hscope. unfold apply_statics_in_scope in Hscope.
    unfold copy_call.
    destruct (ap_call_kind ap) as [id | ]; [ | reflexivity ].
    cbn [among existsb] in Hscope.
    rewrite andb_false_r in Hscope. discriminate.
Qed.

Fixpoint binds_no_static_no_code (e : expr) {struct e} :
  binds_no_static e = true -> binds_no_code (map snd ids) e = true.
Proof.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl; try (intros _; reflexivity).
  - apply binds_no_static_no_code.
  - discriminate.
  - destruct h as [params metadata handler]. simpl. intros Hno.
    apply andb_true_iff in Hno as [Hh Hb].
    rewrite (binds_no_static_no_code handler Hh),
      (binds_no_static_no_code body Hb).
    reflexivity.
  - intros Hno. apply andb_true_iff in Hno as [Hhs Hb].
    rewrite (binds_no_static_no_code body Hb), andb_true_r.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    simpl in Hhs. apply andb_true_iff in Hhs as [Hh Hrest].
    rewrite (binds_no_static_no_code handler Hh), (IH Hrest). reflexivity.
Qed.

(* What may follow a unit is the same in the second state, and binds
   no new id. *)
Lemma observers_copied (slots : list value_slot) (conts : cont_env) :
  observers slots conts ->
  copy_conts conts = conts
  ∧ conts_terms_ok (handler_binds_no_code (map snd ids)) conts.
Proof.
  unfold observers, copy_conts, conts_terms_ok.
  induction conts as [ | [k c] conts IH]; simpl;
    [ intros _; split; [ reflexivity | exact I ] | ].
  intros H. inversion H as [ | kc rest Hc Hrest]; subst.
  destruct (IH Hrest) as [Hsame Hok].
  revert Hc.
  destruct c
    as [ [params metadata body] vars kept recursive | k0 | c0 extra
       | remaining remaining_arity return_arity return_to exn_to ];
    simpl; try contradiction;
    [ | intros _; rewrite Hsame; split; [ reflexivity | ];
        split; [ exact I | exact Hok ] ].
  destruct vars; [ | contradiction ].
  destruct kept; [ | contradiction ].
  destruct recursive; [ | contradiction ].
  intros (Hno & Hscope & _). simpl.
  rewrite Hsame, (copy_expr_nothing body Hno Hscope).
  split; [ reflexivity | ].
  split; [ | exact Hok ].
  split; [ exact (binds_no_static_no_code body Hno) | ].
  split; [ reflexivity | exact I ].
Qed.

End Copied.

(* The static environment of the imports of a unit is such: it holds
   no code, and an import is no closure. *)
Lemma copy_statics_imports (ids : list (code_id * code_id))
    (symbols : list symbol) (imports : list (symbol * import_class)) :
  copy_statics ids symbols (imports_statics imports) = imports_statics imports.
Proof.
  unfold copy_statics, imports_statics.
  cbn [static_code static_values flat_map]. f_equal.
  induction imports as [ | [s c] imports IH]; cbn [map fst snd];
    [ reflexivity | ].
  rewrite IH. unfold copy_allocated. destruct (among s symbols); reflexivity.
Qed.

Lemma copy_static_code_refines_from (ids : list (code_id * code_id))
    (symbols : list symbol) (slots : list value_slot) (e e' : expr)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  copy_static_code ids symbols e = Some e' ->
  observers slots conts ->
  static_code statics = [] ->
  copy_statics ids symbols statics = statics ->
  refines (State e [] conts statics hp rd) (State e' [] conts statics hp rd).
Proof.
  unfold copy_static_code, copy_ok.
  destruct (distinct (map fst ids) && distinct (map snd ids)
            && binds_no_code (map snd ids) e) eqn:Hok; [ | discriminate ].
  apply andb_true_iff in Hok as [Hok Hbinds].
  apply andb_true_iff in Hok as [_ Hnew].
  intros [= <-] Hobservers.
  destruct (observers_copied ids symbols slots conts Hobservers)
    as [Hsame Hconts].
  exact (copy_expr_refines_from ids symbols Hnew e conts statics hp rd_ Hbinds
           Hconts Hsame).
Qed.

Theorem copy_static_code_refines_observed :
  copy_static_code_refines_observed_statement.
Proof.
  intros imports slots ids symbols conts e e' Hcopy Hobservers.
  exact (copy_static_code_refines_from ids symbols slots e e' conts
           (imports_statics imports) empty_heap no_reads Hcopy Hobservers
             eq_refl
           (copy_statics_imports ids symbols imports)).
Qed.

Theorem copy_static_code_refines_imports :
  copy_static_code_refines_imports_statement.
Proof.
  intros imports ids symbols e e' Hcopy. unfold unit_refines.
  exact (copy_static_code_refines_observed imports [] ids symbols [] e e'
           Hcopy (Forall_nil _)).
Qed.

Theorem copy_static_code_program_refines :
  copy_static_code_program_refines_statement.
Proof.
  intros ids symbols e e' Hcopy.
  exact (copy_static_code_refines_imports [] ids symbols e e' Hcopy).
Qed.

(* ======================================================================== *)
(* The copies in turn                                                       *)
(* ======================================================================== *)

Theorem with_copies_none : with_copies_none_statement.
Proof. intros e. reflexivity. Qed.

Theorem with_copies_refines_observed :
  with_copies_refines_observed_statement.
Proof.
  intros imports slots copies conts.
  induction copies as [ | [ids symbols] copies IH]; intros e Hobservers;
    cbn [with_copies]; [ apply refines_refl | ].
  destruct (copy_static_code ids symbols e) as [e' | ] eqn:Hcopy;
    [ | exact (IH e Hobservers) ].
  exact (refines_trans _ _ _
           (copy_static_code_refines_observed imports slots ids symbols conts
              e e' Hcopy Hobservers)
           (IH e' Hobservers)).
Qed.

Theorem with_copies_refines_imports : with_copies_refines_imports_statement.
Proof.
  intros imports copies e. unfold unit_refines.
  exact (with_copies_refines_observed imports [] copies [] e (Forall_nil _)).
Qed.

Theorem with_copies_program_refines : with_copies_program_refines_statement.
Proof.
  intros copies e. exact (with_copies_refines_imports [] copies e).
Qed.

(* ======================================================================== *)
(* The rewrite and the check under renaming                                 *)
(* ======================================================================== *)

(* The three laws are proved from the definitions by the tactic of
   FlambdaEquivarianceTactic.v, which states and proves by itself the
   laws of the functions that the rewrite and the check are made
   of. *)
Theorem copy_expr_equivariant : copy_expr_equivariant_statement.
Proof. equivariance. Qed.

Theorem binds_no_code_equivariant : binds_no_code_equivariant_statement.
Proof. equivariance. Qed.

Theorem with_copies_equivariant : with_copies_equivariant_statement.
Proof. equivariance. Qed.

(* ======================================================================== *)
(* The names that the unit with the copies binds                            *)
(* ======================================================================== *)

(* The rewrite binds the symbols that the term binds, and the code ids
   that it binds among others. *)
Fixpoint copy_expr_symbols (ids : list (code_id * code_id))
    (symbols : list symbol) (e : expr) {struct e} :
  static_names symbols_of_binding (copy_expr ids symbols e)
  = static_names symbols_of_binding e
with copy_handler_symbols (ids : list (code_id * code_id))
    (symbols : list symbol) (h : continuation_handler) {struct h} :
  handler_static_names symbols_of_binding (copy_handler ids symbols h)
  = handler_static_names symbols_of_binding h.
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply copy_expr_symbols.
    + rewrite copy_expr_symbols. f_equal.
      induction bindings as [ | binding bindings IH]; [ reflexivity | ].
      cbn [flat_map]. rewrite flat_map_app, IH. f_equal.
      destruct binding as [id c | id | s const]; cbn [copy_binding];
        [ destruct (lookup id ids) | | ]; reflexivity.
    + rewrite copy_handler_symbols, copy_expr_symbols. reflexivity.
    + rewrite copy_expr_symbols. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite copy_handler_symbols, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl. apply copy_expr_symbols.
Qed.

Fixpoint copy_expr_code_ids (ids : list (code_id * code_id))
    (symbols : list symbol) (e : expr) {struct e} :
  incl (static_names code_ids_of_binding e)
    (static_names code_ids_of_binding (copy_expr ids symbols e))
with copy_handler_code_ids (ids : list (code_id * code_id))
    (symbols : list symbol) (h : continuation_handler) {struct h} :
  incl (handler_static_names code_ids_of_binding h)
    (handler_static_names code_ids_of_binding (copy_handler ids symbols h)).
Proof.
  - destruct e
      as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try apply incl_refl.
    + apply copy_expr_code_ids.
    + apply incl_app_app; [ | apply copy_expr_code_ids ].
      induction bindings as [ | binding bindings IH]; [ apply incl_refl | ].
      cbn [flat_map]. rewrite flat_map_app.
      apply incl_app_app; [ | exact IH ].
      destruct binding as [id c | id | s const]; cbn [copy_binding];
        [ destruct (lookup id ids) | | ]; simpl; try apply incl_refl.
      intros id' [<- | []]. left. reflexivity.
    + apply incl_app_app;
        [ apply copy_handler_code_ids | apply copy_expr_code_ids ].
    + apply incl_app_app; [ | apply copy_expr_code_ids ].
      induction handlers as [ | [k h] handlers IH]; simpl;
        [ apply incl_refl | ].
      apply incl_app_app; [ apply copy_handler_code_ids | exact IH ].
  - destruct h as [params metadata handler]. simpl.
    apply copy_expr_code_ids.
Qed.

(* A static environment that binds no name that the unit with the
   copies binds, binds none that the unit binds. *)
Lemma free_in_under_copies (statics : static_env)
    (copies : list (list (code_id * code_id) * list symbol)) :
  ∀ e : expr,
    static_names_free_in statics (with_copies copies e) = true ->
    static_names_free_in statics e = true.
Proof.
  induction copies as [ | [ids symbols] copies IH]; intros e;
    cbn [with_copies]; [ exact (fun H => H) | ].
  intros H. specialize (IH _ H). revert IH.
  unfold copy_static_code.
  destruct (copy_ok ids e); [ | exact (fun H => H) ].
  unfold static_names_free_in. rewrite copy_expr_symbols.
  intros Hfree. apply andb_true_iff in Hfree as [Hsymbols Hcode].
  rewrite Hsymbols. cbn [andb].
  apply forallb_forall. intros id Hid.
  exact (proj1 (forallb_forall _ _) Hcode id
           (copy_expr_code_ids ids symbols e id Hid)).
Qed.

(* The copies in turn, of a unit that something follows, in the form
   that composes. *)
Lemma with_copies_refines_under (imports : list (symbol * import_class))
    (slots : list value_slot)
    (copies : list (list (code_id * code_id) * list symbol)) (e : expr) :
  refines_under_observers slots (imports_statics imports) empty_heap no_reads e
    (with_copies copies e).
Proof.
  intros conts Hobservers. apply refines_observed.
  exact (with_copies_refines_observed imports slots copies conts e
           Hobservers).
Qed.

(* The unit whose code is copied twice, computed. *)
Theorem with_copies_twice : with_copies_twice_statement.
Proof. reflexivity. Qed.

Theorem with_copies_two_in_one : with_copies_two_in_one_statement.
Proof. split; reflexivity. Qed.

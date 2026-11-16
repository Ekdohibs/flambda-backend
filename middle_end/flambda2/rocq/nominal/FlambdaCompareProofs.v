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

(* Proofs of the statements of FlambdaCompareSpec.v. Nothing here needs
   review.

   Two expressions that [expr_eqb] accepts under some pairs of names, run
   under environments that give paired names the same things, stay in step
   ([matched_step]): a bisimulation
   (FlambdaSemanticsProofs.bisimulation_equivalent). The values of the two
   runs are the same; the continuation closures and the code are not, as
   their bound names differ, so those are related ([cont_matches],
   [statics_match]). The runs start under one pair of environments, which
   needs [expr_eqb] to be reflexive ([expr_eqb_refl]).

   The environments need agree only on the names that are free in the
   expression, and the environments a continuation closure keeps only on
   those free in its handlers ([envs_match] and [conts_match] take the
   names that matter). So the bisimulation also serves where one
   expression runs under two environments that differ elsewhere, and it
   is the only one about bound names: the statements of
   FlambdaNominalSpec.v are proved from it ([matched_equivalent]). *)

From Stdlib Require Import ZArith List Bool Arith.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec FlambdaCompareSpec.
Import ListNotations.

(* ======================================================================== *)
(* Names                                                                    *)
(* ======================================================================== *)

Lemma names_match_nil (a a' : positive) : names_match [] a a' = true -> a = a'.
Proof. apply Pos.eqb_eq. Qed.

Definition reflexive_map (m : name_map) : Prop :=
  ∀ a : positive, names_match m a a = true.

Lemma reflexive_map_nil : reflexive_map [].
Proof. intros a. apply Pos.eqb_refl. Qed.

Lemma reflexive_map_cons (m : name_map) (b : positive) :
  reflexive_map m -> reflexive_map ((b, b) :: m).
Proof.
  intros Hm a. simpl. destruct (Pos.eqb a b); [ reflexivity | apply Hm ].
Qed.

Lemma bind_names_refl (xs : list positive) :
  ∀ m : name_map,
    reflexive_map m ->
    ∃ m' : name_map, bind_names xs xs m = Some m' ∧ reflexive_map m'.
Proof.
  induction xs as [ | x xs IH]; intros m Hm; simpl; [ eauto | ].
  apply IH, reflexive_map_cons, Hm.
Qed.

Lemma combine_names_refl (xs : list positive) (m : name_map) :
  reflexive_map m ->
  ∃ m' : name_map, combine_names xs xs m = Some m' ∧ reflexive_map m'.
Proof.
  intros Hm. unfold combine_names. rewrite Nat.eqb_refl.
  eexists. split; [ reflexivity | ].
  induction xs as [ | x xs IH]; simpl; [ exact Hm | ].
  apply reflexive_map_cons, IH.
Qed.

(* ======================================================================== *)
(* The comparison is reflexive                                              *)
(* ======================================================================== *)

Lemma forall2b_refl {A : Type} (f : A -> A -> bool) (l : list A) :
  (∀ a : A, f a a = true) -> forall2b f l l = true.
Proof.
  intros Hf. induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite Hf, IH. reflexivity.
Qed.

Lemma const_eqb_refl (c : const) : const_eqb c c = true.
Proof. unfold const_eqb. destruct (const_eq_dec c c); congruence. Qed.

Lemma simple_eqb_refl (vm : name_map) (s : simple) :
  reflexive_map vm -> simple_eqb vm s s = true.
Proof.
  intros Hvm. destruct s as [x | sym | c]; simpl;
    [ apply Hvm | apply Pos.eqb_refl | apply const_eqb_refl ].
Qed.

Lemma simples_eqb_refl (vm : name_map) (args : list simple) :
  reflexive_map vm -> simples_eqb vm args args = true.
Proof.
  intros Hvm. apply forall2b_refl. intros s. apply simple_eqb_refl, Hvm.
Qed.

Lemma value_slots_eqb_refl (vm : name_map)
    (slots : list (value_slot * simple)) :
  reflexive_map vm -> value_slots_eqb vm slots slots = true.
Proof.
  intros Hvm. apply forall2b_refl. intros slot.
  rewrite Pos.eqb_refl, simple_eqb_refl; [ reflexivity | exact Hvm ].
Qed.

Lemma names_eqb_refl (l : list (positive * positive)) : names_eqb l l = true.
Proof.
  apply forall2b_refl. intros a. rewrite !Pos.eqb_refl. reflexivity.
Qed.

(* Lists of names that compare the same are the same. *)
Lemma names_eqb_eq (l l' : list (positive * positive)) :
  names_eqb l l' = true -> l = l'.
Proof.
  revert l'.
  induction l as [ | [a b] l IH]; intros [ | [a' b'] l']; simpl;
    try discriminate; [ reflexivity | ].
  intros Heq. apply andb_true_iff in Heq as [Hab Hl].
  apply andb_true_iff in Hab as [Ha Hb].
  apply Pos.eqb_eq in Ha. apply Pos.eqb_eq in Hb. simpl in Ha, Hb.
  subst a' b'. rewrite (IH l' Hl). reflexivity.
Qed.

Lemma named_eqb_refl (vm : name_map) (d : named) :
  reflexive_map vm -> named_eqb vm d d = true.
Proof.
  intros Hvm.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure]; simpl;
    rewrite ?prim_eqb_refl, ?Pos.eqb_refl, ?simple_eqb_refl,
      ?simples_eqb_refl, ?value_slots_eqb_refl, ?names_eqb_refl;
    auto.
Qed.

Lemma static_const_eqb_refl (vm : name_map) (const : static_const) :
  reflexive_map vm -> static_const_eqb vm const const = true.
Proof.
  intros Hvm.
  destruct const
    as [code slots | fslot syms decls slots | tag fields | s | bytes
        | elements]; simpl;
    rewrite ?Pos.eqb_refl, ?Z.eqb_refl, ?simple_eqb_refl, ?simples_eqb_refl,
      ?value_slots_eqb_refl, ?names_eqb_refl;
    auto.
  apply forall2b_refl, Z.eqb_refl.
Qed.

Lemma raise_kind_eqb_refl (kind : option raise_kind) :
  raise_kind_eqb kind kind = true.
Proof. destruct kind as [[ | | ] | ]; reflexivity. Qed.

Lemma trap_action_eqb_refl (km : name_map) (action : option trap_action) :
  reflexive_map km -> trap_action_eqb km action action = true.
Proof.
  intros Hkm. destruct action as [[k | k kind] | ]; simpl;
    rewrite ?Hkm, ?raise_kind_eqb_refl; reflexivity.
Qed.

Lemma apply_cont_eqb_refl (vm km : name_map) (ac : apply_cont) :
  reflexive_map vm -> reflexive_map km -> apply_cont_eqb vm km ac ac = true.
Proof.
  intros Hvm Hkm. unfold apply_cont_eqb.
  rewrite Hkm, simples_eqb_refl, trap_action_eqb_refl;
    [ reflexivity | exact Hkm | exact Hvm ].
Qed.

Lemma call_kind_eqb_refl (kind : call_kind) : call_kind_eqb kind kind = true.
Proof.
  destruct kind as [code | | [code_ids | ]]; simpl;
    [ apply Pos.eqb_refl | reflexivity | | reflexivity ].
  apply forall2b_refl, Pos.eqb_refl.
Qed.

Lemma apply_eqb_refl (vm km : name_map) (ap : apply_expr) :
  reflexive_map vm -> reflexive_map km -> apply_eqb vm km ap ap = true.
Proof.
  intros Hvm Hkm. unfold apply_eqb.
  rewrite !Hkm, simple_eqb_refl, !simples_eqb_refl, kinds_eqb_refl,
    !arity_eqb_refl
    by exact Hvm.
  rewrite call_kind_eqb_refl. reflexivity.
Qed.

Fixpoint expr_eqb_refl (e : expr) :
  ∀ vm km : name_map,
    reflexive_map vm -> reflexive_map km -> expr_eqb vm km e e = true
with handler_eqb_refl (h : continuation_handler) :
  ∀ vm km : name_map,
    reflexive_map vm -> reflexive_map km -> handler_eqb vm km h h = true
with binding_eqb_refl (binding : static_binding) :
  ∀ vm : name_map, reflexive_map vm -> binding_eqb vm binding binding = true
with code_eqb_refl (c : code) : code_eqb c c = true.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros vm km Hvm Hkm; simpl.
    + rewrite kind_eqb_refl, named_eqb_refl by exact Hvm.
      apply expr_eqb_refl; [ apply reflexive_map_cons, Hvm | exact Hkm ].
    + rewrite (expr_eqb_refl body vm km Hvm Hkm), andb_true_r.
      induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
      rewrite (binding_eqb_refl binding vm Hvm), IH. reflexivity.
    + rewrite (handler_eqb_refl h vm km Hvm Hkm).
      apply expr_eqb_refl; [ exact Hvm | apply reflexive_map_cons, Hkm ].
    + destruct (combine_names_refl (map fst handlers) km Hkm)
        as (km' & -> & Hkm').
      rewrite (expr_eqb_refl body vm km' Hvm Hkm'), andb_true_r.
      clear Hkm. induction handlers as [ | [k h] handlers IH]; simpl;
        [ reflexivity | ].
      rewrite (handler_eqb_refl h vm km' Hvm Hkm'), IH. reflexivity.
    + apply apply_eqb_refl; assumption.
    + apply apply_cont_eqb_refl; assumption.
    + rewrite simple_eqb_refl by exact Hvm. apply forall2b_refl. intros arm.
      rewrite Z.eqb_refl, apply_cont_eqb_refl; auto.
    + reflexivity.
  - destruct h as [params metadata handler]. intros vm km Hvm Hkm. simpl.
    destruct (bind_names_refl params vm Hvm) as (vm' & -> & Hvm').
    rewrite handler_metadata_eqb_refl. apply expr_eqb_refl; assumption.
  - destruct binding as [id c | id | sym const]; intros vm Hvm; simpl.
    + rewrite Pos.eqb_refl, (code_eqb_refl c). reflexivity.
    + apply Pos.eqb_refl.
    + rewrite Pos.eqb_refl, static_const_eqb_refl; auto.
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    destruct (bind_names_refl params [(my_closure, my_closure)]
                (reflexive_map_cons _ _ reflexive_map_nil))
      as (vm & -> & Hvm).
    rewrite code_metadata_eqb_refl. apply expr_eqb_refl; [ exact Hvm | ].
    apply reflexive_map_cons, reflexive_map_cons, reflexive_map_nil.
Qed.

(* ======================================================================== *)
(* Environments that give paired names the same things                      *)
(* ======================================================================== *)

(* The names that matter are those that satisfy [relevant]: the free
   variables of what is about to run. *)
Definition envs_match (relevant : variable -> Prop) (vm : name_map)
    (vars vars' : env) : Prop :=
  ∀ x x' : variable,
    relevant x -> names_match vm x x' = true ->
    lookup x vars = lookup x' vars'.

Lemma envs_match_same (relevant : variable -> Prop) (vars : env) :
  envs_match relevant [] vars vars.
Proof.
  intros x x' _ Hx. apply names_match_nil in Hx. subst x'. reflexivity.
Qed.

Lemma envs_match_weaken (relevant fewer : variable -> Prop) (vm : name_map)
    (vars vars' : env) :
  (∀ x : variable, fewer x -> relevant x) ->
  envs_match relevant vm vars vars' ->
  envs_match fewer vm vars vars'.
Proof. intros Hfewer Hmatch x x' Hx. apply Hmatch, Hfewer, Hx. Qed.

(* From the match of two environments, their match on fewer names; what
   is left to show is that the fewer are among the first. *)
Ltac fewer_variables H :=
  refine (envs_match_weaken _ _ _ _ _ _ H); intros ? ?.

Lemma envs_match_cons (relevant inner : variable -> Prop) (vm : name_map)
    (vars vars' : env) (a a' : variable) (v : value) :
  (∀ x : variable, inner x -> x = a ∨ relevant x) ->
  envs_match relevant vm vars vars' ->
  envs_match inner ((a, a') :: vm) ((a, v) :: vars) ((a', v) :: vars').
Proof.
  intros Hinner Hmatch x x' Hx. simpl.
  destruct (Pos.eqb_spec x a) as [-> | Hne].
  - intros ->. reflexivity.
  - destruct (Pos.eqb x' a'); [ discriminate | ]. apply Hmatch.
    destruct (Hinner x Hx); [ contradiction | assumption ].
Qed.

(* Binding paired parameters: both fail or both succeed, and then the
   environments match under the pairs with those of the parameters. *)
Lemma bind_params_match (params : list variable) :
  ∀ (params' : list variable) (vs : list value) (vm vm' : name_map)
    (vars vars' : env) (relevant inner : variable -> Prop),
    (∀ x : variable, inner x -> x ∈ params ∨ relevant x) ->
    bind_names params params' vm = Some vm' ->
    envs_match relevant vm vars vars' ->
    match bind_params params vs vars, bind_params params' vs vars' with
    | Some bound, Some bound' => envs_match inner vm' bound bound'
    | None, None => True
    | _, _ => False
    end.
Proof.
  induction params as [ | p params IH];
    intros [ | p' params'] vs vm vm' vars vars' relevant inner Hinner; simpl;
    try discriminate.
  - intros [= <-] Hmatch. destruct vs; [ | exact I ].
    apply (envs_match_weaken relevant); [ | exact Hmatch ].
    intros x Hx. destruct (Hinner x Hx) as [[] | Hrelevant]. exact Hrelevant.
  - intros Hbind Hmatch. destruct vs as [ | v vs]; [ exact I | ].
    apply (IH params' vs ((p, p') :: vm) vm' ((p, v) :: vars)
             ((p', v) :: vars') (fun x => x = p ∨ relevant x) inner);
      [ | exact Hbind | ].
    + intros x Hx. destruct (Hinner x Hx) as [[<- | Hin] | Hrelevant]; auto.
    + apply (envs_match_cons relevant); [ auto | exact Hmatch ].
Qed.

Lemma eval_simple_match (vm : name_map) (vars vars' : env) (s s' : simple) :
  envs_match (fun x => var_free_in_simple x s) vm vars vars' ->
  simple_eqb vm s s' = true ->
  eval_simple vars s = eval_simple vars' s'.
Proof.
  intros Hmatch.
  destruct s as [x | sym | c], s' as [x' | sym' | c']; simpl;
    try discriminate.
  - apply Hmatch. reflexivity.
  - intros Heq. apply Pos.eqb_eq in Heq. subst sym'. reflexivity.
  - unfold const_eqb. destruct (const_eq_dec c c'); [ | discriminate ].
    subst c'. reflexivity.
Qed.

Lemma eval_simples_match (vm : name_map) (vars vars' : env)
    (args : list simple) :
  envs_match (fun x => var_free_in_simples x args) vm vars vars' ->
  ∀ args' : list simple,
    simples_eqb vm args args' = true ->
    eval_simples vars args = eval_simples vars' args'.
Proof.
  unfold simples_eqb, var_free_in_simples.
  induction args as [ | s args IH]; intros Hmatch [ | s' args']; simpl;
    try discriminate; [ reflexivity | ].
  intros Heq. apply andb_true_iff in Heq as [Hs Hargs].
  assert (Hfirst : envs_match (fun x => var_free_in_simple x s) vm vars vars')
    by (fewer_variables Hmatch; apply Exists_cons_hd; assumption).
  assert (Hrest :
    envs_match (fun x => Exists (var_free_in_simple x) args) vm vars vars')
    by (fewer_variables Hmatch; apply Exists_cons_tl; assumption).
  rewrite (eval_simple_match vm vars vars' s s' Hfirst Hs),
    (IH Hrest args' Hargs).
  reflexivity.
Qed.

Lemma eval_value_slots_match (vm : name_map) (vars vars' : env)
    (slots : list (value_slot * simple)) :
  envs_match
    (fun x => Exists (fun slot => var_free_in_simple x (snd slot)) slots)
    vm vars vars' ->
  ∀ slots' : list (value_slot * simple),
    value_slots_eqb vm slots slots' = true ->
    eval_value_slots vars slots = eval_value_slots vars' slots'.
Proof.
  unfold value_slots_eqb.
  induction slots as [ | [slot s] slots IH];
    intros Hmatch [ | [slot' s'] slots']; simpl; try discriminate;
    [ reflexivity | ].
  intros Heq. apply andb_true_iff in Heq as [Hslot Hslots].
  apply andb_true_iff in Hslot as [Hslot Hs].
  apply Pos.eqb_eq in Hslot. subst slot'.
  assert (Hfirst : envs_match (fun x => var_free_in_simple x s) vm vars vars')
    by (fewer_variables Hmatch; apply Exists_cons_hd; assumption).
  assert (Hrest :
    envs_match
      (fun x => Exists (fun slot => var_free_in_simple x (snd slot)) slots)
      vm vars vars')
    by (fewer_variables Hmatch; apply Exists_cons_tl; assumption).
  rewrite (eval_simple_match vm vars vars' s s' Hfirst Hs),
    (IH Hrest slots' Hslots).
  reflexivity.
Qed.

(* ======================================================================== *)
(* Static environments                                                      *)
(* ======================================================================== *)

(* The same statically allocated values and, for each code id, code that
   is the same up to its bound names. *)
Definition statics_match (statics statics' : static_env) : Prop :=
  Forall2
    (fun ic ic' => fst ic = fst ic' ∧ code_eqb (snd ic) (snd ic') = true)
    (static_code statics) (static_code statics')
  ∧ static_values statics = static_values statics'.

Lemma statics_match_same (statics : static_env) :
  statics_match statics statics.
Proof.
  split; [ | reflexivity ].
  induction (static_code statics) as [ | ic code IH];
    [ apply Forall2_nil | ].
  apply Forall2_cons; [ | exact IH ].
  split; [ reflexivity | apply code_eqb_refl ].
Qed.

Lemma lookup_code_match (id : code_id) (statics statics' : static_env) :
  statics_match statics statics' ->
  match lookup id (static_code statics), lookup id (static_code statics') with
  | Some c, Some c' => code_eqb c c' = true
  | None, None => True
  | _, _ => False
  end.
Proof.
  intros [Hcode _].
  induction Hcode as [ | [id1 c] [id1' c'] code code' [Hid Hc] _ IH]; simpl;
    [ exact I | ].
  simpl in Hid. subst id1'. destruct (Pos.eqb id id1); [ exact Hc | exact IH ].
Qed.

Lemma closure_of_match (statics statics' : static_env) (v : value) :
  statics_match statics statics' ->
  closure_of statics v = closure_of statics' v.
Proof.
  intros [_ Hvalues]. unfold closure_of. rewrite Hvalues. reflexivity.
Qed.

Lemma project_value_slot_match (statics statics' : static_env)
    (project_from : function_slot) (slot : value_slot) (v : value) :
  statics_match statics statics' ->
  project_value_slot statics project_from slot v
  = project_value_slot statics' project_from slot v.
Proof.
  intros [_ Hvalues]. unfold project_value_slot, closure_in_set.
  rewrite Hvalues. reflexivity.
Qed.

Lemma project_function_slot_match (statics statics' : static_env)
    (move_from move_to : function_slot) (v : value) :
  statics_match statics statics' ->
  project_function_slot statics move_from move_to v
  = project_function_slot statics' move_from move_to v.
Proof.
  intros [_ Hvalues]. unfold project_function_slot. rewrite Hvalues.
  reflexivity.
Qed.

Lemma eval_named_match (vm : name_map) (vars vars' : env)
    (statics statics' : static_env) (d d' : named) :
  envs_match (fun x => var_free_in_named x d) vm vars vars' ->
  statics_match statics statics' ->
  named_eqb vm d d' = true ->
  eval_named statics vars d = eval_named statics' vars' d'.
Proof.
  intros Hmatch Hstatics.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure],
    d' as [s' | p' args' | code' slots' | fslot' decls' slots'
          | slot' closure' | pfrom' slot' closure'
          | move_from' move_to' closure'];
    simpl; try discriminate; intros Heq.
  - exact (eval_simple_match vm vars vars' s s' Hmatch Heq).
  - apply andb_true_iff in Heq as [Hp Hargs]. apply prim_eqb_sound in Hp.
    subst p'.
    rewrite (eval_simples_match vm vars vars' args Hmatch args' Hargs).
    destruct (eval_simples vars' args') as [vs | ]; simpl; [ | reflexivity ].
    apply eval_prim_same_static_values, Hstatics.
  - apply andb_true_iff in Heq as [Hcode Hslots]. apply Pos.eqb_eq in Hcode.
    subst code'.
    rewrite (eval_value_slots_match vm vars vars' slots Hmatch slots' Hslots).
    reflexivity.
  - apply andb_true_iff in Heq as [Heq Hslots].
    apply andb_true_iff in Heq as [Hfslot Hdecls].
    apply Pos.eqb_eq in Hfslot. apply names_eqb_eq in Hdecls.
    subst fslot' decls'.
    rewrite (eval_value_slots_match vm vars vars' slots Hmatch slots' Hslots).
    reflexivity.
  - apply andb_true_iff in Heq as [Hslot Hclosure].
    apply Pos.eqb_eq in Hslot. subst slot'.
    rewrite (eval_simple_match vm vars vars' closure closure' Hmatch Hclosure).
    destruct (eval_simple vars' closure') as [v | ]; simpl; [ | reflexivity ].
    rewrite (closure_of_match statics statics' v Hstatics). reflexivity.
  - apply andb_true_iff in Heq as [Heq Hclosure].
    apply andb_true_iff in Heq as [Hfrom Hslot].
    apply Pos.eqb_eq in Hfrom. apply Pos.eqb_eq in Hslot.
    subst pfrom' slot'.
    rewrite (eval_simple_match vm vars vars' closure closure' Hmatch Hclosure).
    destruct (eval_simple vars' closure') as [v | ]; simpl; [ | reflexivity ].
    apply project_value_slot_match, Hstatics.
  - apply andb_true_iff in Heq as [Heq Hclosure].
    apply andb_true_iff in Heq as [Hfrom Hto].
    apply Pos.eqb_eq in Hfrom. apply Pos.eqb_eq in Hto.
    subst move_from' move_to'.
    rewrite (eval_simple_match vm vars vars' closure closure' Hmatch Hclosure).
    destruct (eval_simple vars' closure') as [v | ]; simpl; [ | reflexivity ].
    apply project_function_slot_match, Hstatics.
Qed.

(* And they do the same on a heap. *)
Lemma exec_named_match (vm : name_map) (vars vars' : env)
    (statics statics' : static_env) (hp : heap) (rd : read_record)
      (d d' : named) :
  envs_match (fun x => var_free_in_named x d) vm vars vars' ->
  statics_match statics statics' ->
  named_eqb vm d d' = true ->
  exec_named statics hp rd vars d = exec_named statics' hp rd vars' d'.
Proof.
  intros Hmatch Hstatics Heq. symmetry. apply exec_named_agrees.
  - symmetry.
    exact (eval_named_match vm vars vars' statics statics' d d' Hmatch
             Hstatics Heq).
  - intros p args ->.
    destruct d' as [s' | p' args' | code' slots' | fslot' decls' slots'
          | slot' closure' | pfrom' slot' closure'
          | move_from' move_to' closure'];
      simpl in Heq; try discriminate.
    apply andb_true_iff in Heq as [Hp Hargs]. apply prim_eqb_sound in Hp.
    subst p'.
    exists args'. split; [ reflexivity | split ].
    + symmetry.
      exact (eval_simples_match vm vars vars' args Hmatch args' Hargs).
    + intros vs s _ _. rewrite (proj2 Hstatics). reflexivity.
  - intros p args ->.
    destruct d as [s | p0 args0 | | | | | ]; simpl in Heq; try discriminate.
    eauto.
Qed.

Lemma forall2b_Z_eqb (l l' : list Z) : forall2b Z.eqb l l' = true -> l = l'.
Proof.
  revert l'. induction l as [ | z l IH]; intros [ | z' l']; simpl;
    try discriminate; [ reflexivity | ].
  intros Heq. apply andb_true_iff in Heq as [Hz Hl].
  apply Z.eqb_eq in Hz. rewrite Hz, (IH l' Hl). reflexivity.
Qed.

Lemma eval_static_const_match (vm : name_map) (vars vars' : env)
    (const const' : static_const) :
  envs_match (fun x => var_free_in_static_const x const) vm vars vars' ->
  static_const_eqb vm const const' = true ->
  eval_static_const vars const = eval_static_const vars' const'.
Proof.
  intros Hmatch.
  destruct const
    as [code slots | fslot syms decls slots | tag fields | s | bytes
        | elements],
    const'
    as [code' slots' | fslot' syms' decls' slots' | tag' fields' | s'
       | bytes' | elements'];
    simpl; try discriminate; intros Heq.
  - apply andb_true_iff in Heq as [Hcode Hslots]. apply Pos.eqb_eq in Hcode.
    subst code'.
    rewrite (eval_value_slots_match vm vars vars' slots Hmatch slots' Hslots).
    reflexivity.
  - apply andb_true_iff in Heq as [Heq Hslots].
    apply andb_true_iff in Heq as [Heq Hdecls].
    apply andb_true_iff in Heq as [Hfslot Hsyms].
    apply Pos.eqb_eq in Hfslot. apply names_eqb_eq in Hsyms.
    apply names_eqb_eq in Hdecls. subst fslot' syms' decls'.
    rewrite (eval_value_slots_match vm vars vars' slots Hmatch slots' Hslots).
    reflexivity.
  - apply andb_true_iff in Heq as [Htag Hfields]. apply Z.eqb_eq in Htag.
    subst tag'.
    rewrite (eval_simples_match vm vars vars' fields Hmatch fields' Hfields).
    reflexivity.
  - rewrite (eval_simple_match vm vars vars' s s' Hmatch Heq). reflexivity.
  - rewrite (forall2b_Z_eqb bytes bytes' Heq). reflexivity.
  - rewrite
      (eval_simples_match vm vars vars' elements Hmatch elements' Heq).
    reflexivity.
Qed.

(* Matching groups of bindings added to matching static environments:
   both fail, or the results match. *)
Lemma eval_static_bindings_match (vm : name_map) (vars vars' : env)
    (bindings : list static_binding) :
  envs_match (fun x => Exists (var_free_in_static_binding x) bindings) vm vars
    vars' ->
  ∀ (bindings' : list static_binding) (statics statics' : static_env),
    forall2b (binding_eqb vm) bindings bindings' = true ->
    statics_match statics statics' ->
    match
      eval_static_bindings vars bindings statics,
      eval_static_bindings vars' bindings' statics'
    with
    | Some extended, Some extended' => statics_match extended extended'
    | None, None => True
    | _, _ => False
    end.
Proof.
  induction bindings as [ | binding bindings IH];
    intros Hmatch [ | binding' bindings'] statics statics'; simpl;
    try discriminate; [ intros _ Hstatics; exact Hstatics | ].
  intros Heq Hstatics. pose proof Hstatics as [Hcode Hvalues].
  apply andb_true_iff in Heq as [Hbinding Hrest].
  assert (Hmatch_first :
    envs_match (fun x => var_free_in_static_binding x binding) vm vars vars')
    by (fewer_variables Hmatch; apply Exists_cons_hd; assumption).
  assert (Hmatch_rest :
    envs_match (fun x => Exists (var_free_in_static_binding x) bindings) vm
      vars vars')
    by (fewer_variables Hmatch; apply Exists_cons_tl; assumption).
  specialize (IH Hmatch_rest).
  destruct binding as [id c | id | sym const],
    binding' as [id' c' | id' | sym' const'];
    simpl in Hbinding; try discriminate.
  - apply andb_true_iff in Hbinding as [Hid Hc]. apply Pos.eqb_eq in Hid.
    subst id'.
    pose proof (lookup_code_match id statics statics' Hstatics) as Hid.
    destruct (lookup id (static_code statics)),
      (lookup id (static_code statics')); try contradiction; [ exact I | ].
    apply (IH bindings' _ _ Hrest). split; [ | exact Hvalues ].
    apply Forall2_cons; [ split; [ reflexivity | exact Hc ] | exact Hcode ].
  - apply (IH bindings' _ _ Hrest). split; assumption.
  - apply andb_true_iff in Hbinding as [Hsym Hconst].
    apply Pos.eqb_eq in Hsym. subst sym'. rewrite <- Hvalues.
    destruct (lookup sym (static_values statics)); [ exact I | ].
    rewrite (eval_static_const_match vm vars vars' const const' Hmatch_first
      Hconst).
    destruct (eval_static_const vars' const') as [allocated | ]; simpl;
      [ | exact I ].
    apply (IH bindings' _ _ Hrest). split; [ exact Hcode | ].
    simpl. rewrite Hvalues. reflexivity.
Qed.

(* ======================================================================== *)
(* Continuations                                                            *)
(* ======================================================================== *)

(* The handlers of a recursive group, compared one with one. *)
Definition handlers_eqb (vm km : name_map)
    (handlers handlers' : list (continuation * continuation_handler)) : bool :=
  forall2b (fun '(_, h) '(_, h') => handler_eqb vm km h h') handlers handlers'.

(* What two continuation names that match stand for: the same way out, the
   same continuation, or closures whose handlers are the same up to bound
   names, under the pairs of names of where they were defined, with
   environments that match on the names free in the handlers: those of
   the handler itself and of the handlers defined with it, any of which
   it may come to run; or what is left of the same over-application,
   with continuations that match. *)
#[local] Set Warnings "-register-all".
Inductive cont_matches : cont_value -> cont_value -> Prop :=
  | Same_matches (c : cont_value) : cont_matches c c
  | Closure_matches (vm km km' : name_map) (h h' : continuation_handler)
      (vars vars' : env) (conts conts' : cont_env)
      (recursive recursive' : list (continuation * continuation_handler)) :
    combine_names (map fst recursive) (map fst recursive') km = Some km' ->
    handler_eqb vm km' h h' = true ->
    handlers_eqb vm km' recursive recursive' = true ->
    envs_match
      (fun x =>
         var_free_in_handler x h
         ∨ any_handler (var_free_in_handler x) recursive)
      vm vars vars' ->
    (∀ k k' : continuation,
      (cont_free_in_handler k h
       ∨ any_handler (cont_free_in_handler k) recursive)
      ∧ lookup k recursive = None ->
      names_match km k k' = true ->
      cont_matches (resolve k conts) (resolve k' conts')) ->
    cont_matches (Closure h vars conts recursive)
      (Closure h' vars' conts' recursive')
  | Extra_args_matches (c c' : cont_value) (extra : list value) :
    cont_matches c c' ->
    cont_matches (Extra_args c extra) (Extra_args c' extra)
  | Over_application_matches (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    cont_matches return_to return_to' ->
    cont_matches exn_to exn_to' ->
    cont_matches
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining remaining_arity return_arity return_to'
         exn_to').

Lemma with_extra_args_matches (c c' : cont_value) (extra : list value) :
  cont_matches c c' ->
  cont_matches (with_extra_args c extra) (with_extra_args c' extra).
Proof.
  intros Hc. destruct extra; simpl;
    [ exact Hc | apply Extra_args_matches, Hc ].
Qed.

(* As [envs_match]: the names that matter are the free continuations of
   what is about to run. *)
Definition conts_match (relevant : continuation -> Prop) (km : name_map)
    (conts conts' : cont_env) : Prop :=
  ∀ k k' : continuation,
    relevant k -> names_match km k k' = true ->
    cont_matches (resolve k conts) (resolve k' conts').

Lemma conts_match_same (relevant : continuation -> Prop) (conts : cont_env) :
  conts_match relevant [] conts conts.
Proof.
  intros k k' _ Hk. apply names_match_nil in Hk. subst k'. apply Same_matches.
Qed.

Lemma conts_match_weaken (relevant fewer : continuation -> Prop)
    (km : name_map) (conts conts' : cont_env) :
  (∀ k : continuation, fewer k -> relevant k) ->
  conts_match relevant km conts conts' ->
  conts_match fewer km conts conts'.
Proof. intros Hfewer Hmatch k k' Hk. apply Hmatch, Hfewer, Hk. Qed.

Ltac fewer_continuations H :=
  refine (conts_match_weaken _ _ _ _ _ _ H); intros ? ?.

Lemma conts_match_cons (relevant inner : continuation -> Prop)
    (km : name_map) (conts conts' : cont_env) (a a' : continuation)
    (c c' : cont_value) :
  (∀ k : continuation, inner k -> k = a ∨ relevant k) ->
  conts_match relevant km conts conts' ->
  cont_matches c c' ->
  conts_match inner ((a, a') :: km) ((a, c) :: conts) ((a', c') :: conts').
Proof.
  intros Hinner Hmatch Hc k k' Hk. unfold resolve. simpl.
  destruct (Pos.eqb_spec k a) as [-> | Hne].
  - intros ->. exact Hc.
  - destruct (Pos.eqb k' a'); [ discriminate | ]. apply Hmatch.
    destruct (Hinner k Hk); [ contradiction | assumption ].
Qed.

(* What satisfies a predicate and is one of the handlers. *)
Lemma any_handler_in {A : Type} (P : A -> Prop) (k : continuation) (h : A)
    (handlers : list (continuation * A)) :
  (k, h) ∈ handlers -> P h -> any_handler P handlers.
Proof.
  induction handlers as [ | [k0 h0] handlers IH]; simpl; [ intros [] | ].
  intros [[= -> ->] | Hin] Hh; [ left; exact Hh | right; exact (IH Hin Hh) ].
Qed.

(* A name is bound by a list of handlers or is one of its names. *)
Lemma lookup_none_or_in {A : Type} (k : continuation)
    (handlers : list (continuation * A)) :
  lookup k handlers = None ∨ k ∈ map fst handlers.
Proof.
  destruct (lookup k handlers) as [h | ] eqn:Hlookup; [ right | left; auto ].
  apply lookup_in in Hlookup. apply in_map_iff. exists (k, h). auto.
Qed.

(* The environments in which the handlers of two matching recursive groups
   run. [all] and [all'] are the whole groups, kept apart from the handlers
   still to be gone through. The continuations that matter outside the
   group are [relevant]; among them are those free in the group. *)
Lemma rec_conts_match_aux (relevant : continuation -> Prop)
    (vm km km' : name_map) (vars vars' : env) (conts conts' : cont_env)
    (all all' : list (continuation * continuation_handler)) :
  combine_names (map fst all) (map fst all') km = Some km' ->
  handlers_eqb vm km' all all' = true ->
  envs_match (fun x => any_handler (var_free_in_handler x) all) vm vars vars'
  ->
  (∀ k : continuation,
    any_handler (cont_free_in_handler k) all -> lookup k all = None ->
    relevant k) ->
  conts_match relevant km conts conts' ->
  ∀ (handlers handlers' : list (continuation * continuation_handler))
    (inner : continuation -> Prop),
    (∀ (k : continuation) (h : continuation_handler),
      (k, h) ∈ handlers -> (k, h) ∈ all) ->
    (∀ k : continuation, inner k -> k ∈ map fst handlers ∨ relevant k) ->
    handlers_eqb vm km' handlers handlers' = true ->
    conts_match inner (combine (map fst handlers) (map fst handlers') ++ km)
      (map (fun '(k, h) => (k, Closure h vars conts all)) handlers ++ conts)
      (map (fun '(k, h) => (k, Closure h vars' conts' all')) handlers'
       ++ conts').
Proof.
  intros Hnames Hall Hvars Hgroup Hconts. unfold handlers_eqb.
  induction handlers as [ | [k h] handlers IH];
    intros [ | [k' h'] handlers'] inner Hin Hinner; simpl; try discriminate.
  - intros _. apply (conts_match_weaken relevant); [ | exact Hconts ].
    intros j Hj. destruct (Hinner j Hj) as [[] | Hrelevant]. exact Hrelevant.
  - intros Heq. apply andb_true_iff in Heq as [Hh Hrest].
    apply (conts_match_cons (fun j => j ∈ map fst handlers ∨ relevant j)).
    + intros j Hj. destruct (Hinner j Hj) as [[<- | Hbound] | Hrelevant]; auto.
    + apply (IH handlers'); [ | auto | exact Hrest ].
      intros j hj Hj. apply Hin. right. exact Hj.
    + pose proof (Hin k h (or_introl eq_refl)) as Hh_in.
      apply (Closure_matches vm km km' h h' vars vars' conts conts' all all'
               Hnames Hh Hall).
      * fewer_variables Hvars.
        match goal with
        | Hx : _ ∨ _ |- _ =>
          destruct Hx as [Hx | Hx];
            [ exact (any_handler_in _ k h all Hh_in Hx) | exact Hx ]
        end.
      * intros j j' [Hj Hunbound] Hmatch. apply Hconts; [ | exact Hmatch ].
        apply Hgroup; [ | exact Hunbound ].
        destruct Hj as [Hj | Hj];
          [ exact (any_handler_in _ k h all Hh_in Hj) | exact Hj ].
Qed.

Lemma rec_conts_match (relevant inner : continuation -> Prop)
    (vm km km' : name_map) (vars vars' : env) (conts conts' : cont_env)
    (handlers handlers' : list (continuation * continuation_handler)) :
  combine_names (map fst handlers) (map fst handlers') km = Some km' ->
  handlers_eqb vm km' handlers handlers' = true ->
  envs_match (fun x => any_handler (var_free_in_handler x) handlers) vm vars
    vars' ->
  (∀ k : continuation,
    any_handler (cont_free_in_handler k) handlers ->
    lookup k handlers = None ->
    relevant k) ->
  (∀ k : continuation, inner k -> k ∈ map fst handlers ∨ relevant k) ->
  conts_match relevant km conts conts' ->
  conts_match inner km' (rec_closures handlers vars conts ++ conts)
    (rec_closures handlers' vars' conts' ++ conts').
Proof.
  intros Hnames Hhandlers Hvars Hgroup Hinner Hconts.
  pose proof (rec_conts_match_aux relevant vm km km' vars vars' conts conts'
    handlers handlers' Hnames Hhandlers Hvars Hgroup Hconts handlers
    handlers' inner (fun k h Hin => Hin) Hinner Hhandlers)
    as Hmatch.
  assert (Hkm' :
    km' = combine (map fst handlers) (map fst handlers') ++ km).
  { unfold combine_names in Hnames.
    match type of Hnames with
    | context [if ?same_length then _ else _] => destruct same_length
    end;
      cbv iota in Hnames; [ | discriminate ].
    injection Hnames as Hkm'. symmetry. exact Hkm'. }
  rewrite Hkm'. exact Hmatch.
Qed.

(* ======================================================================== *)
(* States                                                                   *)
(* ======================================================================== *)

Inductive state_matches : state -> state -> Prop :=
  | State_matches (vm km : name_map) (e e' : expr) (vars vars' : env)
      (conts conts' : cont_env) (statics statics' : static_env) (hp : heap)
        (rd : read_record) :
    expr_eqb vm km e e' = true ->
    envs_match (fun x => var_free_in_expr x e) vm vars vars' ->
    conts_match (fun k => cont_free_in_expr k e) km conts conts' ->
    statics_match statics statics' ->
    state_matches (State e vars conts statics hp rd)
      (State e' vars' conts' statics' hp rd).

(* A continuation closure matches itself, by the reflexivity of the
   comparison. *)
Lemma closure_matches_itself (h : continuation_handler) (vars : env)
    (conts : cont_env)
    (recursive : list (continuation * continuation_handler)) :
  ∃ km' : name_map,
    combine_names (map fst recursive) (map fst recursive) [] = Some km'
    ∧ handler_eqb [] km' h h = true
    ∧ handlers_eqb [] km' recursive recursive = true.
Proof.
  destruct (combine_names_refl (map fst recursive) [] reflexive_map_nil)
    as (km' & Hnames & Hkm').
  exists km'. split; [ exact Hnames | ].
  split; [ apply handler_eqb_refl; [ apply reflexive_map_nil | exact Hkm' ] | ].
  unfold handlers_eqb. apply forall2b_refl. intros [k h0].
  apply handler_eqb_refl; [ apply reflexive_map_nil | exact Hkm' ].
Qed.

(* Lists of code ids that compare the same are the same. *)
Lemma code_ids_eqb_eq (ids ids' : list code_id) :
  forall2b Pos.eqb ids ids' = true -> ids = ids'.
Proof.
  revert ids'.
  induction ids as [ | id ids IH]; intros [ | id' ids']; simpl;
    try discriminate; [ reflexivity | ].
  intros Heq. apply andb_true_iff in Heq as [Hid Hids].
  apply Pos.eqb_eq in Hid. subst id'. rewrite (IH ids' Hids). reflexivity.
Qed.

Lemma call_kind_eqb_eq (kind kind' : call_kind) :
  call_kind_eqb kind kind' = true -> kind = kind'.
Proof.
  destruct kind as [code | | [ids | ]], kind' as [code' | | [ids' | ]]; simpl;
    try discriminate; try reflexivity.
  - intros Heq. apply Pos.eqb_eq in Heq. subst code'. reflexivity.
  - intros Heq. rewrite (code_ids_eqb_eq ids ids' Heq). reflexivity.
Qed.

(* Expressions that are the same up to bound names agree on whether they
   bind something static. *)
Fixpoint expr_eqb_binds_statics (e : expr) :
  ∀ (vm km : name_map) (e' : expr),
    expr_eqb vm km e e' = true -> binds_statics e = binds_statics e'.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros vm km
      [ x' kind' d' body' | bindings' body' | k' h' body' | handlers' body'
      | ap' | ac' | scrutinee' arms' | ];
    simpl; try discriminate; try reflexivity; intros Heq.
  - apply andb_true_iff in Heq as [_ Hbody].
    exact (expr_eqb_binds_statics body _ _ _ Hbody).
  - apply andb_true_iff in Heq as [Hbindings Hbody].
    rewrite (expr_eqb_binds_statics body _ _ _ Hbody). f_equal.
    clear Hbody. revert bindings' Hbindings.
    induction bindings as [ | binding bindings IH];
      intros [ | binding' bindings']; simpl; try discriminate;
      [ reflexivity | ].
    intros Heq. apply andb_true_iff in Heq as [Hbinding Hrest].
    rewrite (IH _ Hrest). f_equal.
    destruct binding, binding'; simpl in Hbinding; try discriminate;
      reflexivity.
  - apply andb_true_iff in Heq as [Hh Hbody].
    destruct h as [params metadata handler],
      h' as [params' metadata' handler'].
    simpl in Hh |- *.
    destruct (bind_names params params' vm) as [vm' | ]; [ | discriminate ].
    apply andb_true_iff in Hh as [_ Hh].
    rewrite (expr_eqb_binds_statics handler _ _ _ Hh),
      (expr_eqb_binds_statics body _ _ _ Hbody).
    reflexivity.
  - destruct (combine_names (map fst handlers) (map fst handlers') km)
      as [km' | ]; [ | discriminate ].
    apply andb_true_iff in Heq as [Hhandlers Hbody].
    rewrite (expr_eqb_binds_statics body _ _ _ Hbody). f_equal.
    clear Hbody. revert handlers' Hhandlers.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      intros [ | [k' [params' metadata' handler']] handlers']; simpl;
      try discriminate; [ reflexivity | ].
    intros Heq. apply andb_true_iff in Heq as [Hh Hrest].
    destruct (bind_names params params' vm) as [vm' | ]; [ | discriminate ].
    apply andb_true_iff in Hh as [_ Hh].
    rewrite (expr_eqb_binds_statics handler _ _ _ Hh), (IH _ Hrest).
    reflexivity.
Qed.

Lemma handlers_eqb_bind_statics (vm km : name_map)
    (handlers : list (continuation * continuation_handler)) :
  ∀ handlers' : list (continuation * continuation_handler),
    forall2b (fun '(_, h) '(_, h') => handler_eqb vm km h h') handlers
      handlers'
    = true ->
    handlers_bind_statics handlers = handlers_bind_statics handlers'.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH];
    intros [ | [k' [params' metadata' handler']] handlers']; simpl;
    try discriminate; [ reflexivity | ].
  intros Heq. apply andb_true_iff in Heq as [Hh Hrest].
  destruct (bind_names params params' vm) as [vm' | ]; [ | discriminate ].
  apply andb_true_iff in Hh as [_ Hh].
  rewrite (expr_eqb_binds_statics handler _ _ _ Hh), (IH _ Hrest).
  reflexivity.
Qed.

(* Matching static environments bind a code id alike. *)
Lemma codes_agree_match (statics statics' : static_env) (called : code_id) :
  statics_match statics statics' ->
  codes_agree (fun c c' => code_eqb c c' = true) statics statics' called.
Proof.
  intros Hstatics. unfold codes_agree.
  pose proof (lookup_code_match called statics statics' Hstatics) as Hc.
  destruct (lookup called (static_code statics))
    as [[params my_closure ret exn metadata body] | ],
    (lookup called (static_code statics'))
    as [[params' my_closure' ret' exn' metadata' body'] | ];
    try contradiction; [ | exact I ].
  split; [ exact Hc | ]. simpl in Hc |- *.
  destruct (bind_names params params' [(my_closure, my_closure')]);
    [ | discriminate ].
  apply andb_true_iff in Hc as [Hmetadata _].
  apply code_metadata_eqb_eq in Hmetadata. subst metadata'.
  repeat split.
Qed.

(* The same full call of codes that are the same up to bound names,
   returning to matching continuations. *)
Lemma call_code_match (closure : value) (c c' : code) (vs : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  code_eqb c c' = true ->
  statics_match statics statics' ->
  cont_matches return_to return_to' ->
  cont_matches exn_to exn_to' ->
  result_related state_matches
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure c' vs args_arity return_arity return_to' exn_to'
       statics' hp rd).
Proof.
  intros Hc Hstatics Hreturn Hexn.
  destruct c as [params my_closure ret exn metadata body],
    c' as [params' my_closure' ret' exn' metadata' body'].
  unfold call_code. simpl in Hc |- *.
  destruct (bind_names params params' [(my_closure, my_closure')])
    as [vm | ] eqn:Hbind; [ | discriminate ].
  apply andb_true_iff in Hc as [Hmetadata Hc].
  apply code_metadata_eqb_eq in Hmetadata. subst metadata'.
  rewrite <- (expr_eqb_binds_statics body _ _ _ Hc).
  destruct (binds_statics body); [ exact I | ].
  destruct
    (negb
       (kinds_eqb (map kind_of_value vs) (unarize args_arity)
        && kinds_per_parameter_eqb (unarize_per_parameter args_arity)
             (unarize_per_parameter (cm_params_arity metadata))
        && arity_eqb return_arity (cm_result_arity metadata)));
    [ exact I | ].
  pose proof (bind_params_match params params' vs _ vm
    [(my_closure, closure)] [(my_closure', closure)] (fun _ => True)
    (fun x => var_free_in_expr x body) (fun x _ => or_intror I) Hbind
    (envs_match_cons (fun _ => True) (fun _ => True) [] [] [] my_closure
       my_closure' closure (fun x _ => or_intror I)
       (envs_match_same _ [])))
    as Hbound.
  destruct (bind_params params vs [(my_closure, closure)]),
    (bind_params params' vs [(my_closure', closure)]);
    try contradiction; simpl; [ | exact I ].
  apply (State_matches vm [(ret, ret'); (exn, exn')]);
    [ exact Hc | exact Hbound | | exact Hstatics ].
  apply (conts_match_cons (fun _ => True)); [ auto | | exact Hreturn ].
  apply (conts_match_cons (fun _ => True));
    [ auto | apply conts_match_same | exact Hexn ].
Qed.

(* Calls of the same shape, given that what is done with a partial
   application matches. *)
Lemma run_call_match (return_partial return_partial' : value -> step_result)
    (shape shape' : option call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  shapes_related (fun c c' => code_eqb c c' = true) shape shape' ->
  statics_match statics statics' ->
  cont_matches return_to return_to' ->
  cont_matches exn_to exn_to' ->
  (∀ partial : value,
    result_related state_matches (return_partial partial)
      (return_partial' partial)) ->
  result_related state_matches
    (run_call return_partial shape return_arity return_to exn_to statics hp rd)
    (run_call return_partial' shape' return_arity return_to' exn_to'
       statics' hp rd).
Proof.
  intros Hshapes Hstatics Hreturn Hexn Hpartial.
  destruct Hshapes
    as [ | closure c c' vs args_arity Hc
       | closure c c' vs remaining args_arity remaining_arity Hc | partial];
    simpl.
  - exact I.
  - apply call_code_match; assumption.
  - assert (Hresult : code_result_arity c' = code_result_arity c).
    { destruct c as [params my_closure ret exn metadata body],
        c' as [params' my_closure' ret' exn' metadata' body'].
      simpl in Hc |- *.
      destruct (bind_names params params' [(my_closure, my_closure')]);
        [ | discriminate ].
      apply andb_true_iff in Hc as [Hmetadata _].
      apply code_metadata_eqb_eq in Hmetadata. subst metadata'.
      reflexivity. }
    rewrite Hresult.
    apply call_code_match; try assumption.
    apply Over_application_matches; assumption.
  - apply Hpartial.
Qed.

(* Applying matching continuations to the same values, under matching
   static environments. *)
Lemma apply_cont_value_match (c c' : cont_value) (vs : list value)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  cont_matches c c' ->
  statics_match statics statics' ->
  result_related state_matches (apply_cont_value c vs statics hp rd)
    (apply_cont_value c' vs statics' hp rd).
Proof.
  intros Hc Hstatics. revert c' vs Hc.
  assert (Hclosure :
    ∀ (vs : list value) (vm km km' : name_map)
      (h h' : continuation_handler)
      (vars vars' : env) (conts conts' : cont_env)
      (recursive recursive' : list (continuation * continuation_handler)),
      combine_names (map fst recursive) (map fst recursive') km = Some km' ->
      handler_eqb vm km' h h' = true ->
      handlers_eqb vm km' recursive recursive' = true ->
      envs_match
        (fun x =>
           var_free_in_handler x h
           ∨ any_handler (var_free_in_handler x) recursive)
        vm vars vars' ->
      conts_match
        (fun k =>
           (cont_free_in_handler k h
            ∨ any_handler (cont_free_in_handler k) recursive)
           ∧ lookup k recursive = None)
        km conts conts' ->
      result_related state_matches
        (apply_cont_value (Closure h vars conts recursive) vs statics hp rd)
        (apply_cont_value (Closure h' vars' conts' recursive') vs statics' hp
           rd)).
  { intros vs vm km km' [params metadata handler]
      [params' metadata' handler'] vars vars' conts conts' recursive
      recursive' Hnames Hh Hrec Hvars Hconts.
    simpl in Hh |- *.
    destruct (bind_names params params' vm) as [vm' | ] eqn:Hbind;
      [ | discriminate ].
    apply andb_true_iff in Hh as [Hmetadata Hh].
    apply handler_metadata_eqb_eq in Hmetadata. subst metadata'.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    pose proof (bind_params_match params params' vs vm vm' vars vars' _
      (fun x => var_free_in_expr x handler)
      (fun x Hx =>
         match in_dec Pos.eq_dec x params with
         | left Hin => or_introl Hin
         | right Hnotin => or_intror (or_introl (conj Hnotin Hx))
         end)
      Hbind Hvars) as Hbound.
    destruct (bind_params params vs vars), (bind_params params' vs vars');
      try contradiction; simpl; [ | exact I ].
    apply (State_matches vm' km');
      [ exact Hh | exact Hbound | | exact Hstatics ].
    refine
      (rec_conts_match _ _ vm km km' vars vars' conts conts' recursive
         recursive' Hnames Hrec _ _ _ Hconts).
    - fewer_variables Hvars. right. assumption.
    - intros k Hk Hunbound. split; [ right; exact Hk | exact Hunbound ].
    - intros k Hk.
      destruct (lookup_none_or_in k recursive) as [Hunbound | Hin];
        [ right | left; exact Hin ].
      split; [ left; exact Hk | exact Hunbound ]. }
  assert (Hover :
    ∀ (vs : list value) (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value),
      (∀ (c' : cont_value) (vs' : list value),
        cont_matches return_to c' ->
        result_related state_matches
          (apply_cont_value return_to vs' statics hp rd)
          (apply_cont_value c' vs' statics' hp rd)) ->
      cont_matches return_to return_to' ->
      cont_matches exn_to exn_to' ->
      result_related state_matches
        (apply_cont_value
           (Over_application remaining remaining_arity return_arity return_to
              exn_to)
           vs statics hp rd)
        (apply_cont_value
           (Over_application remaining remaining_arity return_arity
              return_to' exn_to')
           vs statics' hp rd)).
  { intros vs remaining remaining_arity return_arity return_to return_to'
      exn_to exn_to' IHreturn Hreturn Hexn.
    rewrite !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ exact I | | exact I ].
    unfold call_function. apply run_call_match; try assumption.
    - apply shape_of_call_related;
        [ symmetry; apply closure_of_match; exact Hstatics | | ].
      + intros called slots _. apply codes_agree_match. exact Hstatics.
      + intros tuple rest p _. symmetry.
        exact (eval_prim_same_static_values _ _ p [tuple] (proj2 Hstatics)).
    - intros partial. apply IHreturn. exact Hreturn. }
  induction c
    as [h vars conts recursive | exit | c IHc extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros c' vs Hc.
  - inversion Hc
      as [c0 | vm km km' h0 h' vars0 vars' conts0 conts' recursive0
                 recursive' Hnames Hh Hrec Hvars Hconts | | ];
      subst.
    + destruct (closure_matches_itself h vars conts recursive)
        as (km' & Hnames & Hh & Hrec).
      exact (Hclosure vs [] [] km' h h vars vars conts conts recursive
        recursive Hnames Hh Hrec (envs_match_same _ vars)
        (conts_match_same _ conts)).
    + exact (Hclosure vs vm km km' h h' vars vars' conts conts' recursive
        recursive' Hnames Hh Hrec Hvars Hconts).
  - inversion Hc; subst. simpl. auto.
  - inversion Hc; subst; simpl; apply IHc;
      [ apply Same_matches | assumption ].
  - inversion Hc
      as [c0 | |
         | remaining0 remaining_arity0 return_arity0 return_to0 return_to'
             exn_to0 exn_to' Hreturn Hexn];
      subst.
    + apply Hover; [ exact IHreturn | apply Same_matches .. ].
    + apply Hover; assumption.
Qed.

(* The same call under matching static environments, returning to matching
   continuations. *)
Lemma call_function_match (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  statics_match statics statics' ->
  cont_matches return_to return_to' ->
  cont_matches exn_to exn_to' ->
  result_related state_matches
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd)
    (call_function callee kind vs args_arity return_arity return_to' exn_to'
       statics' hp rd).
Proof.
  intros Hstatics Hreturn Hexn. unfold call_function.
  apply run_call_match; try assumption.
  - apply shape_of_call_related;
      [ symmetry; apply closure_of_match; exact Hstatics | | ].
    + intros called slots _. apply codes_agree_match. exact Hstatics.
    + intros tuple rest p _. symmetry.
      exact (eval_prim_same_static_values _ _ p [tuple] (proj2 Hstatics)).
  - intros partial. apply apply_cont_value_match; assumption.
Qed.

Lemma find_arm_match (vm km : name_map) (i : Z)
    (arms : list (Z * apply_cont)) :
  ∀ arms' : list (Z * apply_cont),
    forall2b
      (fun arm arm' =>
         Z.eqb (fst arm) (fst arm')
         && apply_cont_eqb vm km (snd arm) (snd arm'))
      arms arms'
    = true ->
    match find_arm i arms, find_arm i arms' with
    | Some ac, Some ac' => apply_cont_eqb vm km ac ac' = true
    | None, None => True
    | _, _ => False
    end.
Proof.
  induction arms as [ | [d ac] arms IH]; intros [ | [d' ac'] arms']; simpl;
    try discriminate; [ intros _; exact I | ].
  intros Heq. apply andb_true_iff in Heq as [Harm Hrest].
  apply andb_true_iff in Harm as [Hd Hac]. apply Z.eqb_eq in Hd. subst d'.
  destruct (Z.eqb i d); [ exact Hac | exact (IH arms' Hrest) ].
Qed.

(* Matching states step alike. *)
Lemma matched_step (s s' : state) :
  state_matches s s' -> result_related state_matches (step s) (step s').
Proof.
  intros [vm km e e' vars vars' conts conts' statics statics' hp rd He Hvars
            Hconts Hstatics].
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ],
    e'
    as [ x' kind' d' body' | bindings' body' | k' h' body' | handlers' body'
       | ap' | ac' | scrutinee' arms' | ];
    simpl in He; try discriminate; try rewrite !step_let_exec;
    autorewrite with step; simpl.
  - (* Let *)
    apply andb_true_iff in He as [Hd Hbody].
    apply andb_true_iff in Hd as [Hkind Hd].
    apply kind_eqb_eq in Hkind. subst kind'.
    assert (Hvars_d :
      envs_match (fun y => var_free_in_named y d) vm vars vars')
      by (fewer_variables Hvars; left; assumption).
    rewrite (exec_named_match vm vars vars' statics statics' hp rd d d' Hvars_d
      Hstatics Hd).
    destruct (exec_named statics' hp rd vars' d') as [[[v hp'] rd'] | ]; simpl;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind); simpl; [ | exact I ].
    apply (State_matches ((x, x') :: vm) km);
      [ exact Hbody | | exact Hconts | exact Hstatics ].
    refine (envs_match_cons _ _ _ _ _ _ _ _ _ Hvars). intros y Hy.
    destruct (Pos.eq_dec y x) as [Heq | Hne]; [ left; exact Heq | ].
    right. right. split; assumption.
  - (* Let_static *)
    apply andb_true_iff in He as [Hbindings Hbody].
    assert (Hvars_bindings :
      envs_match (fun y => Exists (var_free_in_static_binding y) bindings) vm
        vars vars')
      by (fewer_variables Hvars; left; assumption).
    pose proof (eval_static_bindings_match vm vars vars' bindings
      Hvars_bindings bindings' statics statics' Hbindings Hstatics)
      as Hextended.
    destruct (eval_static_bindings vars bindings statics),
      (eval_static_bindings vars' bindings' statics');
      try contradiction; simpl; [ | exact I ].
    apply (State_matches vm km);
      [ exact Hbody | | exact Hconts | exact Hextended ].
    fewer_variables Hvars. right. assumption.
  - (* Let_cont_non_recursive *)
    apply andb_true_iff in He as [Hh Hbody].
    apply (State_matches vm ((k, k') :: km));
      [ exact Hbody | fewer_variables Hvars; right; assumption | 
      | exact Hstatics ].
    refine (conts_match_cons _ _ _ _ _ _ _ _ _ _ Hconts _).
    + intros j Hj.
      destruct (Pos.eq_dec j k) as [Heq | Hne]; [ left; exact Heq | ].
      right. right. split; assumption.
    + apply (Closure_matches vm km km h h' vars vars' conts conts' [] []);
        [ reflexivity | exact Hh | reflexivity | | ].
      * fewer_variables Hvars.
        match goal with
        | Hy : _ ∨ _ |- _ => destruct Hy as [Hy | []]; left; exact Hy
        end.
      * intros j j' [[Hj | []] _]. apply Hconts. left. exact Hj.
  - (* Let_cont_recursive *)
    destruct (combine_names (map fst handlers) (map fst handlers') km)
      as [km' | ] eqn:Hnames; [ | discriminate ].
    apply andb_true_iff in He as [Hhandlers Hbody].
    rewrite <- (handlers_eqb_bind_statics vm km' handlers handlers'
      Hhandlers).
    destruct (handlers_bind_statics handlers); [ exact I | ].
    apply (State_matches vm km');
      [ exact Hbody | fewer_variables Hvars; right; assumption | 
      | exact Hstatics ].
    refine
      (rec_conts_match _ _ vm km km' vars vars' conts conts' handlers
         handlers' Hnames Hhandlers _ _ _ Hconts).
    + fewer_variables Hvars. left. assumption.
    + intros j Hj Hunbound. split; [ exact Hunbound | left; exact Hj ].
    + intros j Hj.
      destruct (lookup_none_or_in j handlers) as [Hunbound | Hbound];
        [ right | left; exact Hbound ].
      split; [ exact Hunbound | right; exact Hj ].
  - (* Apply *)
    unfold apply_eqb in He.
    apply andb_true_iff in He as [He Hreturn_arity].
    apply andb_true_iff in He as [He Hargs_arity].
    apply arity_eqb_eq in Hargs_arity, Hreturn_arity.
    apply andb_true_iff in He as [He _].
    apply andb_true_iff in He as [He Hextra].
    apply andb_true_iff in He as [He Hexn].
    apply andb_true_iff in He as [He Hreturn].
    apply andb_true_iff in He as [He Hargs].
    apply andb_true_iff in He as [Hcallee Hkind].
    apply call_kind_eqb_eq in Hkind.
    assert (Hvars_callee :
      envs_match (fun y => var_free_in_simple y (ap_callee ap)) vm vars vars')
      by (fewer_variables Hvars; left; assumption).
    assert (Hvars_args :
      envs_match (fun y => var_free_in_simples y (ap_args ap)) vm vars vars')
      by (fewer_variables Hvars; right; left; assumption).
    assert (Hvars_extra :
      envs_match (fun y => var_free_in_simples y (exn_extra_simples ap)) vm
        vars vars')
      by (fewer_variables Hvars; right; right; assumption).
    rewrite (eval_simple_match vm vars vars' _ _ Hvars_callee Hcallee),
      (eval_simples_match vm vars vars' _ Hvars_args _ Hargs),
      (eval_simples_match vm vars vars' _ Hvars_extra _ Hextra), <- Hkind,
      <- Hargs_arity, <- Hreturn_arity.
    destruct (eval_simple vars' (ap_callee ap')) as [callee | ];
      [ | exact I ].
    destruct (eval_simples vars' (ap_args ap')) as [vs | ]; [ | exact I ].
    destruct (eval_simples vars' (exn_extra_simples ap')) as [extra | ];
      [ | exact I ].
    apply call_function_match;
      [ exact Hstatics
      | exact (Hconts _ _ (or_introl eq_refl) Hreturn)
      | apply with_extra_args_matches;
        exact (Hconts _ _ (or_intror eq_refl) Hexn) ].
  - (* Apply_cont *)
    unfold apply_cont_eqb in He. apply andb_true_iff in He as [He Htrap].
    apply andb_true_iff in He as [Hk Hargs].
    rewrite (eval_simples_match vm vars vars' _ Hvars _ Hargs).
    destruct (eval_simples vars' (ac_args ac')) as [vs | ]; [ | exact I ].
    apply apply_cont_value_match;
      [ exact (Hconts _ _ (or_introl eq_refl) Hk) | exact Hstatics ].
  - (* Switch *)
    apply andb_true_iff in He as [Hscrutinee Harms].
    assert (Hvars_scrutinee :
      envs_match (fun y => var_free_in_simple y scrutinee) vm vars vars')
      by (fewer_variables Hvars; left; assumption).
    rewrite (eval_simple_match vm vars vars' _ _ Hvars_scrutinee Hscrutinee).
    destruct (eval_simple vars' scrutinee') as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    pose proof (find_arm_match vm km i arms arms' Harms) as Harm.
    destruct (find_arm i arms) as [ac | ] eqn:Hfind,
      (find_arm i arms') as [ac' | ];
      try contradiction; simpl; [ | exact I ].
    apply find_arm_in in Hfind.
    apply (State_matches vm km); [ exact Harm | | | exact Hstatics ].
    + fewer_variables Hvars. right. apply Exists_exists.
      exists (i, ac). split; assumption.
    + fewer_continuations Hconts. apply Exists_exists.
      exists (i, ac). split; assumption.
  - (* Invalid *)
    exact I.
Qed.

(* So two states that match are equivalent. *)
Theorem matched_equivalent (s s' : state) :
  state_matches s s' -> equivalent s s'.
Proof. apply (bisimulation_equivalent state_matches matched_step). Qed.

Theorem expr_eqb_sound : expr_eqb_sound_statement.
Proof.
  intros e e' Heq vars conts statics hp rd. apply matched_equivalent.
  apply (State_matches [] []);
    [ exact Heq | apply envs_match_same | apply conts_match_same
    | apply statics_match_same ].
Qed.

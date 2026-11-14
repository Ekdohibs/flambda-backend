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

(* Proofs of the statements of FlambdaNominalSpec.v. Nothing here needs
   review.

   All of them come from one fact ([renamed_equivalent]): a state and its
   renaming, under environments that agree on the free names along the
   renaming, are equivalent. That fact is an instance of the bisimulation
   of FlambdaCompareProofs.v, which relates two states whose expressions
   the comparison accepts: the comparison accepts an expression against
   its renaming, under the maps of names that pair each name with its
   image ([rename_expr_eqb], [variables_map]). At code the map is that of
   the binders of the code alone, and that the code is closed says that
   its body has no other free name. There is no bisimulation here. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveSpec FlambdaPrimitiveProofs
  FlambdaBindersSpec FlambdaBindersProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaNominalSpec FlambdaFreeNamesProofs
  FlambdaCompareSpec FlambdaCompareProofs.
Import ListNotations.

(* ======================================================================== *)
(* Permutations                                                             *)
(* ======================================================================== *)

Lemma swap_injective (a b x y : positive) : swap a b x = swap a b y -> x = y.
Proof.
  unfold swap.
  destruct (Pos.eqb_spec x a), (Pos.eqb_spec x b), (Pos.eqb_spec y a),
    (Pos.eqb_spec y b); congruence.
Qed.

Lemma permute_injective (p : permutation) (x y : positive) :
  permute p x = permute p y -> x = y.
Proof.
  induction p as [ | [a b] p IH]; simpl; [ auto | ].
  intros Heq. apply IH, (swap_injective a b), Heq.
Qed.

Lemma permute_eqb (p : permutation) (x y : positive) :
  Pos.eqb (permute p x) (permute p y) = Pos.eqb x y.
Proof.
  destruct (Pos.eqb_spec x y) as [-> | Hne]; [ apply Pos.eqb_refl | ].
  apply Pos.eqb_neq. intros Heq. apply Hne, (permute_injective p), Heq.
Qed.

Lemma rename_variable_eqb (r : renaming) (x y : variable) :
  Pos.eqb (rename_variable r x) (rename_variable r y) = Pos.eqb x y.
Proof. apply permute_eqb. Qed.

Lemma rename_continuation_eqb (r : renaming) (k k' : continuation) :
  Pos.eqb (rename_continuation r k) (rename_continuation r k') = Pos.eqb k k'.
Proof. apply permute_eqb. Qed.

(* ======================================================================== *)
(* The identity renaming                                                    *)
(* ======================================================================== *)

Lemma rename_simple_id (s : simple) : rename_simple identity_renaming s = s.
Proof. destruct s; reflexivity. Qed.

Lemma rename_simples_id (args : list simple) :
  map (rename_simple identity_renaming) args = args.
Proof.
  induction args as [ | s args IH]; simpl; [ reflexivity | ].
  rewrite rename_simple_id, IH. reflexivity.
Qed.

Lemma rename_value_slots_id (slots : list (value_slot * simple)) :
  map (fun slot => (fst slot, rename_simple identity_renaming (snd slot)))
    slots
  = slots.
Proof.
  induction slots as [ | [slot s] slots IH]; simpl; [ reflexivity | ].
  rewrite rename_simple_id, IH. reflexivity.
Qed.

Lemma rename_named_id (d : named) : rename_named identity_renaming d = d.
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure]; simpl;
    rewrite ?rename_simple_id, ?rename_simples_id, ?rename_value_slots_id;
    reflexivity.
Qed.

Lemma rename_static_const_id (const : static_const) :
  rename_static_const identity_renaming const = const.
Proof.
  destruct const
    as [code slots | fslot syms decls slots | tag fields | s | bytes
        | elements];
    simpl;
    rewrite ?rename_simple_id, ?rename_simples_id, ?rename_value_slots_id;
    reflexivity.
Qed.

Lemma rename_variables_id (xs : list variable) :
  map (rename_variable identity_renaming) xs = xs.
Proof.
  induction xs as [ | x xs IH]; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Lemma rename_exn_extra_args_id (extra : list (simple * flambda_kind)) :
  map (fun arg => (rename_simple identity_renaming (fst arg), snd arg))
    extra
  = extra.
Proof.
  induction extra as [ | [s kind] extra IH]; simpl; [ reflexivity | ].
  rewrite rename_simple_id, IH. reflexivity.
Qed.

(* The extra arguments of a renamed call are those of the call,
   renamed. *)
Lemma exn_extra_simples_rename (r : renaming) (ap : apply_expr) :
  exn_extra_simples (rename_apply r ap)
  = map (rename_simple r) (exn_extra_simples ap).
Proof. unfold exn_extra_simples. simpl. rewrite !map_map. reflexivity. Qed.

Lemma rename_apply_id (ap : apply_expr) :
  rename_apply identity_renaming ap = ap.
Proof.
  destruct ap. unfold rename_apply. simpl.
  rewrite rename_simple_id, rename_simples_id, rename_exn_extra_args_id.
  reflexivity.
Qed.

Lemma rename_trap_action_id (action : option trap_action) :
  option_map (rename_trap_action identity_renaming) action = action.
Proof. destruct action as [[k | k kind] | ]; reflexivity. Qed.

Lemma rename_apply_cont_id (ac : apply_cont) :
  rename_apply_cont identity_renaming ac = ac.
Proof.
  destruct ac. unfold rename_apply_cont. simpl.
  rewrite rename_simples_id, rename_trap_action_id. reflexivity.
Qed.

(* By induction on the expression, its handlers and its code included. *)
Fixpoint rename_expr_id (e : expr) : rename_expr identity_renaming e = e
with rename_handler_id (h : continuation_handler) :
  rename_handler identity_renaming h = h
with rename_static_binding_id (binding : static_binding) :
  rename_static_binding identity_renaming binding = binding
with rename_code_id (c : code) : rename_code identity_renaming c = c.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl.
    + rewrite rename_named_id, rename_expr_id. reflexivity.
    + rewrite rename_expr_id. f_equal.
      induction bindings as [ | binding bindings IH]; simpl;
        [ reflexivity | ].
      rewrite rename_static_binding_id, IH. reflexivity.
    + rewrite rename_handler_id, rename_expr_id. reflexivity.
    + rewrite rename_expr_id. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite rename_handler_id, IH. reflexivity.
    + rewrite rename_apply_id. reflexivity.
    + rewrite rename_apply_cont_id. reflexivity.
    + rewrite rename_simple_id. f_equal.
      induction arms as [ | [i ac] arms IH]; simpl; [ reflexivity | ].
      rewrite rename_apply_cont_id, IH. reflexivity.
    + reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite rename_expr_id, rename_variables_id. reflexivity.
  - destruct binding as [id c | id | sym const]; simpl.
    + rewrite rename_code_id. reflexivity.
    + reflexivity.
    + rewrite rename_static_const_id. reflexivity.
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    rewrite rename_expr_id, rename_variables_id. reflexivity.
Qed.

(* ======================================================================== *)
(* Lookups under a renaming                                                 *)
(* ======================================================================== *)

Lemma lookup_rename_handlers (r : renaming) (k : continuation)
    (handlers : list (continuation * continuation_handler)) :
  lookup (rename_continuation r k) (rename_handlers r handlers)
  = option_map (rename_handler r) (lookup k handlers).
Proof.
  induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
  rewrite rename_continuation_eqb.
  destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

(* A renaming leaves what binds something static as it is. *)
Fixpoint rename_binds_statics (r : renaming) (e : expr) :
  binds_statics (rename_expr r e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl; try reflexivity.
  - apply rename_binds_statics.
  - rewrite rename_binds_statics. f_equal.
    induction bindings as [ | [id c | id | sym const] bindings IH]; simpl;
      congruence.
  - destruct h as [params metadata handler]. simpl.
    rewrite !rename_binds_statics. reflexivity.
  - rewrite rename_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite rename_binds_statics, IH. reflexivity.
Qed.

(* An expression that binds nothing static binds no code, so that all the
   code in it is closed. *)
Fixpoint binds_statics_code_closed (e : expr) :
  binds_statics e = false -> code_closed e = true
with handler_binds_statics_code_closed (h : continuation_handler) :
  handler_binds_statics h = false -> handler_code_closed h = true.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply binds_statics_code_closed.
    + intros Hstatics. apply orb_false_iff in Hstatics.
      destruct Hstatics as [Hbindings Hbody].
      rewrite (binds_statics_code_closed body Hbody), andb_true_r.
      induction bindings as [ | [id c | id | sym const] bindings IH];
        simpl in *;
        [ reflexivity | discriminate | exact (IH Hbindings) | discriminate ].
    + intros Hstatics. apply orb_false_iff in Hstatics.
      destruct Hstatics as [Hh Hbody].
      rewrite (handler_binds_statics_code_closed h Hh),
        (binds_statics_code_closed body Hbody).
      reflexivity.
    + intros Hstatics. apply orb_false_iff in Hstatics.
      destruct Hstatics as [Hhandlers Hbody].
      rewrite (binds_statics_code_closed body Hbody), andb_true_r.
      induction handlers as [ | [k h] handlers IH]; simpl in *;
        [ reflexivity | ].
      apply orb_false_iff in Hhandlers. destruct Hhandlers as [Hh Hrest].
      rewrite (handler_binds_statics_code_closed h Hh), (IH Hrest).
      reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply binds_statics_code_closed.
Qed.

(* ======================================================================== *)
(* Names paired with their images                                           *)
(* ======================================================================== *)

Section Paired.

Variable f : positive -> positive.

Hypothesis f_eqb : ∀ x y : positive, Pos.eqb (f x) (f y) = Pos.eqb x y.

Definition paired (xs : list positive) : name_map :=
  map (fun x => (x, f x)) xs.

(* A name matches its image if it is one of the paired names, or if it
   did so before they were added. *)
Lemma names_match_paired (xs : list positive) (m : name_map)
    (y : positive) :
  y ∈ xs ∨ names_match m y (f y) = true ->
  names_match (paired xs ++ m) y (f y) = true.
Proof.
  induction xs as [ | x xs IH]; simpl.
  - intros [[] | Hm]. exact Hm.
  - intros Hy. rewrite f_eqb. destruct (Pos.eqb_spec y x) as [-> | Hne];
      [ reflexivity | ].
    apply IH. destruct Hy as [[Heq | Hin] | Hm];
      [ congruence | left; exact Hin | right; exact Hm ].
Qed.

(* And what a name matches is its image, unless it is not one of the
   paired names. *)
Lemma names_match_paired_inv (xs : list positive) (m : name_map)
    (y y' : positive) :
  names_match (paired xs ++ m) y y' = true ->
  y' = f y ∨ (y ∉ xs ∧ names_match m y y' = true).
Proof.
  induction xs as [ | x xs IH]; simpl.
  - intros Hm. right. split; [ intros [] | exact Hm ].
  - destruct (Pos.eqb_spec y x) as [-> | Hne].
    + intros Heq. apply Pos.eqb_eq in Heq. left. exact Heq.
    + destruct (Pos.eqb y' (f x)); [ discriminate | ].
      intros Hmatch.
      destruct (IH Hmatch) as [Heq | [Hnotin Hm]]; [ left; exact Heq | ].
      right. split; [ | exact Hm ].
      intros [Heq | Hin]; [ congruence | contradiction ].
Qed.

Lemma bind_names_paired (xs : list positive) :
  ∀ m : name_map,
    bind_names xs (map f xs) m = Some (paired (rev xs) ++ m).
Proof.
  induction xs as [ | x xs IH]; intros m; simpl; [ reflexivity | ].
  rewrite IH. unfold paired. rewrite map_app, <- app_assoc. reflexivity.
Qed.

Lemma combine_names_paired (xs : list positive) (m : name_map) :
  combine_names xs (map f xs) m = Some (paired xs ++ m).
Proof.
  unfold combine_names. rewrite length_map, Nat.eqb_refl. do 2 f_equal.
  induction xs as [ | x xs IH]; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

End Paired.

Lemma forall2b_map {A B : Type} (f : A -> B -> bool) (g : A -> B)
    (l : list A) :
  (∀ a : A, a ∈ l -> f a (g a) = true) -> forall2b f l (map g l) = true.
Proof.
  induction l as [ | a l IH]; simpl; intros Hf; [ reflexivity | ].
  rewrite (Hf a (or_introl eq_refl)), IH; [ reflexivity | ].
  intros b Hb. apply Hf. right. exact Hb.
Qed.

Lemma forall2b_app {A B : Type} (f : A -> B -> bool) (l1 l2 : list A)
    (l1' l2' : list B) :
  forall2b f l1 l1' = true -> forall2b f l2 l2' = true ->
  forall2b f (l1 ++ l2) (l1' ++ l2') = true.
Proof.
  revert l1'. induction l1 as [ | a l1 IH]; intros [ | b l1']; simpl;
    try discriminate; [ auto | ].
  intros H1 H2. apply andb_true_iff in H1 as [Hab H1].
  rewrite Hab, IH; auto.
Qed.

(* ======================================================================== *)
(* The comparison accepts a renaming                                        *)
(* ======================================================================== *)

Section Renamed.

Variable r : renaming.

Variable vm : name_map.

(* The variables that the map pairs with their images. *)
Definition matched (free : variable -> Prop) : Prop :=
  ∀ x : variable, free x -> names_match vm x (rename_variable r x) = true.

Lemma simple_eqb_renamed (s : simple) :
  matched (fun x => var_free_in_simple x s) ->
  simple_eqb vm s (rename_simple r s) = true.
Proof.
  destruct s as [x | sym | c]; simpl; intros Hfree;
    [ apply Hfree; reflexivity | apply Pos.eqb_refl | apply const_eqb_refl ].
Qed.

Lemma simples_eqb_renamed (args : list simple) :
  matched (fun x => var_free_in_simples x args) ->
  simples_eqb vm args (map (rename_simple r) args) = true.
Proof.
  intros Hfree. apply forall2b_map. intros s Hin. apply simple_eqb_renamed.
  intros x Hx. apply Hfree, Exists_exists. exists s. split; assumption.
Qed.

Lemma value_slots_eqb_renamed (slots : list (value_slot * simple)) :
  matched
    (fun x => Exists (fun slot => var_free_in_simple x (snd slot)) slots) ->
  value_slots_eqb vm slots
    (map (fun slot => (fst slot, rename_simple r (snd slot))) slots)
  = true.
Proof.
  intros Hfree. apply forall2b_map. intros slot Hin. simpl.
  rewrite Pos.eqb_refl. apply simple_eqb_renamed.
  intros x Hx. apply Hfree, Exists_exists. exists slot. split; assumption.
Qed.

Lemma named_eqb_renamed (d : named) :
  matched (fun x => var_free_in_named x d) ->
  named_eqb vm d (rename_named r d) = true.
Proof.
  destruct d
    as [s | p args | code slots | fslot decls slots | slot closure
       | pfrom slot closure | move_from move_to closure];
    simpl; intros Hfree;
    rewrite ?prim_eqb_refl, ?Pos.eqb_refl, ?names_eqb_refl; simpl.
  - apply simple_eqb_renamed, Hfree.
  - apply simples_eqb_renamed, Hfree.
  - apply value_slots_eqb_renamed, Hfree.
  - apply value_slots_eqb_renamed, Hfree.
  - apply simple_eqb_renamed, Hfree.
  - apply simple_eqb_renamed, Hfree.
  - apply simple_eqb_renamed, Hfree.
Qed.

Lemma static_const_eqb_renamed (const : static_const) :
  matched (fun x => var_free_in_static_const x const) ->
  static_const_eqb vm const (rename_static_const r const) = true.
Proof.
  destruct const
    as [code slots | fslot syms decls slots | tag fields | s | bytes
        | elements];
    simpl; intros Hfree;
    rewrite ?Pos.eqb_refl, ?Z.eqb_refl, ?names_eqb_refl; simpl.
  - apply value_slots_eqb_renamed, Hfree.
  - apply value_slots_eqb_renamed, Hfree.
  - apply simples_eqb_renamed, Hfree.
  - apply simple_eqb_renamed, Hfree.
  - apply forall2b_refl, Z.eqb_refl.
  - apply simples_eqb_renamed, Hfree.
Qed.

Lemma apply_cont_eqb_renamed (km : name_map) (ac : apply_cont) :
  matched (fun x => var_free_in_simples x (ac_args ac)) ->
  (∀ k : continuation,
    k ∈ apply_cont_continuations ac ->
    names_match km k (rename_continuation r k) = true) ->
  apply_cont_eqb vm km ac (rename_apply_cont r ac) = true.
Proof.
  intros Hfree Hk. unfold apply_cont_eqb. simpl.
  rewrite (Hk (ac_continuation ac) (or_introl eq_refl)).
  rewrite simples_eqb_renamed by exact Hfree. simpl.
  unfold apply_cont_continuations in Hk.
  destruct (ac_trap_action ac) as [[k | k kind] | ]; simpl;
    [ | | reflexivity ].
  - apply Hk. right. left. reflexivity.
  - rewrite (Hk k (or_intror (or_introl eq_refl))).
    destruct kind as [[ | | ] | ]; reflexivity.
Qed.

Lemma apply_eqb_renamed (km : name_map) (ap : apply_expr) :
  matched
    (fun x =>
       var_free_in_simple x (ap_callee ap)
       ∨ var_free_in_simples x (ap_args ap)
       ∨ var_free_in_simples x (exn_extra_simples ap)) ->
  names_match km (ap_continuation ap)
    (rename_continuation r (ap_continuation ap))
  = true ->
  names_match km (ap_exn_continuation ap)
    (rename_continuation r (ap_exn_continuation ap))
  = true ->
  apply_eqb vm km ap (rename_apply r ap) = true.
Proof.
  intros Hfree Hreturn Hexn. unfold apply_eqb.
  rewrite exn_extra_simples_rename. simpl.
  rewrite Hreturn, Hexn, simple_eqb_renamed, !simples_eqb_renamed.
  - rewrite map_map. simpl. rewrite kinds_eqb_refl.
    destruct (ap_call_kind ap); simpl;
      rewrite ?Pos.eqb_refl, !arity_eqb_refl; reflexivity.
  - intros x Hx. apply Hfree. right. right. exact Hx.
  - intros x Hx. apply Hfree. right. left. exact Hx.
  - intros x Hx. apply Hfree. left. exact Hx.
Qed.

End Renamed.

Lemma map_fst_rename_handlers (r : renaming)
    (handlers : list (continuation * continuation_handler)) :
  map fst (rename_handlers r handlers)
  = map (rename_continuation r) (map fst handlers).
Proof.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

(* The comparison accepts an expression against its renaming, under maps
   that pair each free name with its image, where the code of the
   expression is closed. At a binder the bound names are added to the
   map, each with its image; at code the map is that of its binders. *)
Fixpoint rename_expr_eqb (r : renaming) (e : expr) :
  ∀ vm km : name_map,
    code_closed e = true ->
    matched r vm (fun x => var_free_in_expr x e) ->
    (∀ k : continuation,
      cont_free_in_expr k e ->
      names_match km k (rename_continuation r k) = true) ->
    expr_eqb vm km e (rename_expr r e) = true
with rename_handler_eqb (r : renaming) (h : continuation_handler) :
  ∀ vm km : name_map,
    handler_code_closed h = true ->
    matched r vm (fun x => var_free_in_handler x h) ->
    (∀ k : continuation,
      cont_free_in_handler k h ->
      names_match km k (rename_continuation r k) = true) ->
    handler_eqb vm km h (rename_handler r h) = true
with rename_binding_eqb (r : renaming) (binding : static_binding) :
  ∀ vm : name_map,
    binding_code_closed binding = true ->
    matched r vm (fun x => var_free_in_static_binding x binding) ->
    binding_eqb vm binding (rename_static_binding r binding) = true
with rename_code_eqb (r : renaming) (c : code) :
  code_is_closed c = true -> code_eqb c (rename_code r c) = true.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros vm km Hclosed Hvars Hconts; simpl; simpl in Hclosed.
    + (* Let *)
      rewrite named_eqb_renamed
        by (intros y Hy; apply Hvars; left; exact Hy).
      rewrite (proj2 (kind_eqb_eq kind kind) eq_refl). simpl.
      apply rename_expr_eqb; [ exact Hclosed | | exact Hconts ].
      intros y Hy.
      apply (names_match_paired (rename_variable r) (rename_variable_eqb r)
               [x] vm y).
      destruct (Pos.eq_dec y x) as [-> | Hne];
        [ left; left; reflexivity | right ].
      apply Hvars. right. split; assumption.
    + (* Let_static *)
      apply andb_true_iff in Hclosed. destruct Hclosed as [Hbindings Hbody].
      apply andb_true_iff. split.
      * assert (Hfree :
          matched r vm
            (fun x => Exists (var_free_in_static_binding x) bindings))
          by (intros x Hx; apply Hvars; left; exact Hx).
        clear Hvars Hconts Hbody.
        induction bindings as [ | binding bindings IH]; simpl;
          [ reflexivity | ].
        simpl in Hbindings. apply andb_true_iff in Hbindings.
        destruct Hbindings as [Hfirst Hrest].
        rewrite (rename_binding_eqb r binding vm Hfirst), IH;
          [ reflexivity | exact Hrest | | ].
        -- intros x Hx. apply Hfree, Exists_cons_tl, Hx.
        -- intros x Hx. apply Hfree, Exists_cons_hd, Hx.
      * apply rename_expr_eqb; [ exact Hbody | | exact Hconts ].
        intros x Hx. apply Hvars. right. exact Hx.
    + (* Let_cont_non_recursive *)
      apply andb_true_iff in Hclosed. destruct Hclosed as [Hh Hbody].
      rewrite (rename_handler_eqb r h vm km Hh);
        [ | intros x Hx; apply Hvars; left; exact Hx
          | intros k' Hk'; apply Hconts; left; exact Hk' ].
      apply rename_expr_eqb; [ exact Hbody | | ].
      * intros x Hx. apply Hvars. right. exact Hx.
      * intros k' Hk'.
        apply (names_match_paired (rename_continuation r)
                 (rename_continuation_eqb r) [k] km k').
        destruct (Pos.eq_dec k' k) as [-> | Hne];
          [ left; left; reflexivity | right ].
        apply Hconts. right. split; assumption.
    + (* Let_cont_recursive *)
      apply andb_true_iff in Hclosed. destruct Hclosed as [Hhandlers Hbody].
      fold (rename_handlers r handlers).
      rewrite map_fst_rename_handlers,
        (combine_names_paired (rename_continuation r)).
      unfold rename_handlers.
      remember (paired (rename_continuation r) (map fst handlers) ++ km)
        as km' eqn:Hkm'.
      assert (Hconts' :
        ∀ k : continuation,
          any_handler (cont_free_in_handler k) handlers
          ∨ cont_free_in_expr k body ->
          names_match km' k (rename_continuation r k) = true).
      { intros k Hk. subst km'.
        apply names_match_paired; [ apply rename_continuation_eqb | ].
        destruct (lookup_none_or_in k handlers) as [Hunbound | Hbound];
          [ right | left; exact Hbound ].
        apply Hconts. split; assumption. }
      apply andb_true_iff. split.
      * assert (Hfree :
          matched r vm
            (fun x => any_handler (var_free_in_handler x) handlers))
          by (intros x Hx; apply Hvars; left; exact Hx).
        assert (Hfree_conts :
          ∀ k : continuation,
            any_handler (cont_free_in_handler k) handlers ->
            names_match km' k (rename_continuation r k) = true)
          by (intros k Hk; apply Hconts'; left; exact Hk).
        clear Hvars Hconts Hconts' Hkm' Hbody.
        induction handlers as [ | [k h] handlers IH]; simpl;
          [ reflexivity | ].
        simpl in Hhandlers. apply andb_true_iff in Hhandlers.
        destruct Hhandlers as [Hfirst Hrest].
        rewrite (rename_handler_eqb r h vm km' Hfirst), IH;
          [ reflexivity | exact Hrest | | | | ].
        -- intros x Hx. apply Hfree. right. exact Hx.
        -- intros k' Hk'. apply Hfree_conts. right. exact Hk'.
        -- intros x Hx. apply Hfree. left. exact Hx.
        -- intros k' Hk'. apply Hfree_conts. left. exact Hk'.
      * apply rename_expr_eqb; [ exact Hbody | | ].
        -- intros x Hx. apply Hvars. right. exact Hx.
        -- intros k Hk. apply Hconts'. right. exact Hk.
    + (* Apply *)
      apply apply_eqb_renamed; [ exact Hvars | | ]; apply Hconts; simpl;
        [ left | right ]; reflexivity.
    + (* Apply_cont *)
      apply apply_cont_eqb_renamed; [ exact Hvars | exact Hconts ].
    + (* Switch *)
      rewrite simple_eqb_renamed
        by (intros x Hx; apply Hvars; left; exact Hx).
      apply forall2b_map. intros arm Hin. simpl. rewrite Z.eqb_refl.
      apply apply_cont_eqb_renamed.
      * intros x Hx. apply Hvars. right. apply Exists_exists.
        exists arm. split; assumption.
      * intros k Hk. apply Hconts, Exists_exists. exists arm.
        split; assumption.
    + (* Invalid *)
      reflexivity.
  - destruct h as [params metadata handler].
    intros vm km Hclosed Hvars Hconts.
    simpl. simpl in Hclosed. rewrite handler_metadata_eqb_refl.
    rewrite (bind_names_paired (rename_variable r)).
    apply rename_expr_eqb; [ exact Hclosed | | exact Hconts ].
    intros x Hx.
    apply names_match_paired; [ apply rename_variable_eqb | ].
    destruct (in_dec Pos.eq_dec x params) as [Hin | Hnotin];
      [ left; apply in_rev in Hin; exact Hin | right ].
    apply Hvars. split; assumption.
  - destruct binding as [id c | id | sym const]; intros vm Hclosed Hvars;
      simpl; simpl in Hclosed; rewrite Pos.eqb_refl; [ | reflexivity | ].
    + apply rename_code_eqb, Hclosed.
    + apply static_const_eqb_renamed, Hvars.
  - intros Hclosed.
    destruct (proj1 (closed_code c) Hclosed) as [Hnames Hbody].
    destruct c as [params my_closure ret exn metadata body].
    simpl in Hnames, Hbody.
    unfold names_among in Hnames. apply andb_true_iff in Hnames.
    destruct Hnames as [Hvars Hconts]. simpl.
    rewrite (bind_names_paired (rename_variable r)).
    rewrite code_metadata_eqb_refl. cbn [andb].
    apply rename_expr_eqb; [ exact Hbody | | ].
    + intros x Hx.
      apply names_match_paired; [ apply rename_variable_eqb | ].
      apply (var_free_agrees_expr x body Hbody) in Hx.
      pose proof (all_among_among _ _ x Hvars Hx) as Hamong.
      apply among_In in Hamong.
      destruct Hamong as [<- | Hin];
        [ right | left; apply in_rev in Hin; exact Hin ].
      apply (names_match_paired (rename_variable r) (rename_variable_eqb r)
               [my_closure] [] my_closure).
      left. left. reflexivity.
    + intros k Hk.
      apply (names_match_paired (rename_continuation r)
               (rename_continuation_eqb r) [ret; exn] [] k).
      left. apply (cont_free_agrees_expr k body Hbody) in Hk.
      apply among_In. exact (all_among_among _ _ k Hconts Hk).
Qed.

(* ======================================================================== *)
(* A renaming as a map of names                                             *)
(* ======================================================================== *)

(* The names that a permutation may move. *)
Fixpoint moved (p : permutation) : list positive :=
  match p with
  | [] => []
  | (a, b) :: p => a :: b :: moved p
  end.

Lemma permute_not_moved (p : permutation) (x : positive) :
  x ∉ moved p -> permute p x = x.
Proof.
  induction p as [ | [a b] p IH]; simpl; [ reflexivity | ].
  intros Hnotin.
  rewrite IH by (intros Hin; apply Hnotin; right; right; exact Hin).
  unfold swap.
  destruct (Pos.eqb_spec x a) as [-> | _];
    [ exfalso; apply Hnotin; left; reflexivity | ].
  destruct (Pos.eqb_spec x b) as [-> | _];
    [ exfalso; apply Hnotin; right; left; reflexivity | reflexivity ].
Qed.

(* The map that pairs every name with its image: the names moved are
   paired, and one that is not is its own image. *)
Lemma names_match_graph (p : permutation) (x x' : positive) :
  names_match (paired (permute p) (moved p)) x x' = true <-> x' = permute p x.
Proof.
  rewrite <- (app_nil_r (paired (permute p) (moved p))). split.
  - intros Hmatch.
    apply names_match_paired_inv in Hmatch as [Heq | [Hnotin Heq]];
      [ exact Heq | ].
    apply names_match_nil in Heq. subst x'. symmetry.
    apply permute_not_moved, Hnotin.
  - intros ->. apply names_match_paired; [ apply permute_eqb | ].
    destruct (in_dec Pos.eq_dec x (moved p)) as [Hin | Hnotin];
      [ left; exact Hin | right ].
    simpl. rewrite (permute_not_moved p x Hnotin). apply Pos.eqb_refl.
Qed.

Definition variables_map (r : renaming) : name_map :=
  paired (rename_variable r) (moved (variable_permutation r)).

Definition continuations_map (r : renaming) : name_map :=
  paired (rename_continuation r) (moved (continuation_permutation r)).

Lemma variables_map_graph (r : renaming) (x x' : variable) :
  names_match (variables_map r) x x' = true <-> x' = rename_variable r x.
Proof. exact (names_match_graph (variable_permutation r) x x'). Qed.

Lemma continuations_map_graph (r : renaming) (k k' : continuation) :
  names_match (continuations_map r) k k' = true
  <-> k' = rename_continuation r k.
Proof. exact (names_match_graph (continuation_permutation r) k k'). Qed.

(* ======================================================================== *)
(* A state and its renaming                                                 *)
(* ======================================================================== *)

(* Two continuation environments agree at [k] along [r]: what [k] stands
   for in the one and what [r] sends it to stands for in the other match
   (FlambdaCompareProofs.cont_matches). If neither binds it, [r] fixes
   it, as it is observed. *)
Definition conts_agree_at (r : renaming) (k : continuation)
    (conts conts' : cont_env) : Prop :=
  cont_matches (resolve k conts) (resolve (rename_continuation r k) conts').

(* One environment agrees with itself at a continuation the renaming
   fixes. *)
Lemma conts_agree_same (r : renaming) (k : continuation) (conts : cont_env) :
  rename_continuation r k = k -> conts_agree_at r k conts conts.
Proof.
  intros Hfix. unfold conts_agree_at. rewrite Hfix. apply Same_matches.
Qed.

(* A state and its renaming, under environments that agree on its free
   names along the renaming: the comparison accepts the two expressions
   under the maps of the renaming, and the environments match under
   them. *)
Theorem renamed_equivalent (r : renaming) (e : expr) (vars vars' : env)
    (conts conts' : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  code_closed e = true ->
  (∀ x : variable,
    var_free_in_expr x e ->
    lookup (rename_variable r x) vars' = lookup x vars) ->
  (∀ k : continuation,
    cont_free_in_expr k e -> conts_agree_at r k conts conts') ->
  equivalent (State e vars conts statics hp rd)
    (State (rename_expr r e) vars' conts' statics hp rd).
Proof.
  intros Hclosed Hvars Hconts. apply matched_equivalent.
  apply (State_matches (variables_map r) (continuations_map r)).
  - apply rename_expr_eqb; [ exact Hclosed | | ].
    + intros x _. apply variables_map_graph. reflexivity.
    + intros k _. apply continuations_map_graph. reflexivity.
  - intros x x' Hx Hmatch. apply variables_map_graph in Hmatch. subst x'.
    symmetry. apply Hvars, Hx.
  - intros k k' Hk Hmatch. apply continuations_map_graph in Hmatch.
    subst k'. apply Hconts, Hk.
  - apply statics_match_same.
Qed.

(* One expression under environments that agree on its free names. No
   name is renamed, and nothing is asked of its code. *)
Lemma coincide_equivalent (e : expr) (vars vars' : env)
    (conts conts' : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  (∀ x : variable, var_free_in_expr x e -> lookup x vars' = lookup x vars) ->
  (∀ k : continuation,
    cont_free_in_expr k e -> conts_agree_at identity_renaming k conts conts')
  ->
  equivalent (State e vars conts statics hp rd)
    (State e vars' conts' statics hp rd).
Proof.
  intros Hvars Hconts. apply matched_equivalent.
  apply (State_matches (variables_map identity_renaming)
           (continuations_map identity_renaming)).
  - apply expr_eqb_refl; apply reflexive_map_nil.
  - intros x x' Hx Hmatch. apply variables_map_graph in Hmatch. subst x'.
    symmetry. apply Hvars, Hx.
  - intros k k' Hk Hmatch. apply continuations_map_graph in Hmatch.
    subst k'. apply Hconts, Hk.
  - apply statics_match_same.
Qed.

(* ======================================================================== *)
(* Binders are name abstractions                                            *)
(* ======================================================================== *)

(* Each statement is the soundness of the comparison
   (FlambdaCompareProofs.expr_eqb_sound) at two expressions that it
   accepts. *)

(* A renaming that fixes the free names: each matches its image under
   maps that match every name with itself. *)
Lemma fixed_matched (r : renaming) (vm : name_map)
    (free : variable -> Prop) :
  reflexive_map vm ->
  (∀ x : variable, free x -> rename_variable r x = x) ->
  matched r vm free.
Proof. intros Hvm Hfixed x Hx. rewrite (Hfixed x Hx). apply Hvm. Qed.

Lemma fixed_conts_matched (r : renaming) (km : name_map)
    (free : continuation -> Prop) :
  reflexive_map km ->
  (∀ k : continuation, free k -> rename_continuation r k = k) ->
  ∀ k : continuation,
    free k -> names_match km k (rename_continuation r k) = true.
Proof. intros Hkm Hfixed k Hk. rewrite (Hfixed k Hk). apply Hkm. Qed.

Lemma rename_expr_accepted (r : renaming) (e : expr) :
  code_closed e = true ->
  (∀ x : variable, var_free_in_expr x e -> rename_variable r x = x) ->
  (∀ k : continuation,
    cont_free_in_expr k e -> rename_continuation r k = k) ->
  expr_eqb [] [] e (rename_expr r e) = true.
Proof.
  intros Hclosed Hvars Hconts. apply rename_expr_eqb; [ exact Hclosed | | ].
  - apply fixed_matched; [ apply reflexive_map_nil | exact Hvars ].
  - apply (fixed_conts_matched r [] _ reflexive_map_nil Hconts).
Qed.

Lemma rename_let_accepted (r : renaming) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  code_closed body = true ->
  (∀ y : variable,
    y <> x -> var_free_in_expr y body -> rename_variable r y = y) ->
  (∀ k : continuation,
    cont_free_in_expr k body -> rename_continuation r k = k) ->
  expr_eqb [] [] (Let x kind d body)
    (Let (rename_variable r x) kind d (rename_expr r body))
  = true.
Proof.
  intros Hclosed Hvars Hconts. simpl.
  rewrite (proj2 (kind_eqb_eq kind kind) eq_refl).
  rewrite named_eqb_refl by apply reflexive_map_nil. cbn [andb].
  apply rename_expr_eqb;
    [ exact Hclosed |
    | apply (fixed_conts_matched r [] _ reflexive_map_nil Hconts) ].
  intros y Hy.
  apply (names_match_paired (rename_variable r) (rename_variable_eqb r)
           [x] [] y).
  destruct (Pos.eq_dec y x) as [-> | Hne];
    [ left; left; reflexivity | right ].
  simpl. rewrite (Hvars y Hne Hy). apply Pos.eqb_refl.
Qed.

Lemma rename_let_cont_accepted (r : renaming) (k : continuation)
    (h : continuation_handler) (body : expr) :
  code_closed body = true ->
  (∀ y : variable, var_free_in_expr y body -> rename_variable r y = y) ->
  (∀ k' : continuation,
    k' <> k -> cont_free_in_expr k' body -> rename_continuation r k' = k')
  ->
  expr_eqb [] [] (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive (rename_continuation r k) h (rename_expr r body))
  = true.
Proof.
  intros Hclosed Hvars Hconts. simpl.
  rewrite handler_eqb_refl by apply reflexive_map_nil.
  apply rename_expr_eqb;
    [ exact Hclosed
    | apply fixed_matched; [ apply reflexive_map_nil | exact Hvars ] | ].
  intros k' Hk'.
  apply (names_match_paired (rename_continuation r)
           (rename_continuation_eqb r) [k] [] k').
  destruct (Pos.eq_dec k' k) as [-> | Hne];
    [ left; left; reflexivity | right ].
  simpl. rewrite (Hconts k' Hne Hk'). apply Pos.eqb_refl.
Qed.

Lemma rename_handler_params_accepted (r : renaming) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler body : expr) :
  code_closed handler = true ->
  (∀ y : variable,
    y ∉ params -> var_free_in_expr y handler -> rename_variable r y = y) ->
  (∀ k' : continuation,
    cont_free_in_expr k' handler -> rename_continuation r k' = k') ->
  expr_eqb [] []
    (Let_cont_non_recursive k (Handler params metadata handler) body)
    (Let_cont_non_recursive k
       (rename_handler r (Handler params metadata handler)) body)
  = true.
Proof.
  intros Hclosed Hvars Hconts.
  change
    (handler_eqb [] [] (Handler params metadata handler)
       (rename_handler r (Handler params metadata handler))
     && expr_eqb [] [(k, k)] body body
     = true).
  rewrite (rename_handler_eqb r (Handler params metadata handler) [] []
             Hclosed).
  - apply expr_eqb_refl;
      [ apply reflexive_map_nil
      | apply reflexive_map_cons, reflexive_map_nil ].
  - apply fixed_matched; [ apply reflexive_map_nil | ].
    intros x [Hnotin Hx]. exact (Hvars x Hnotin Hx).
  - apply (fixed_conts_matched r [] _ reflexive_map_nil Hconts).
Qed.

(* The parameters of one handler of a recursive group: that handler is
   renamed by [r] and the others are left as they are. *)
Lemma rename_recursive_handler_params_accepted (r : renaming)
    (before : list (continuation * continuation_handler))
    (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler : expr)
    (after : list (continuation * continuation_handler)) (body : expr) :
  code_closed handler = true ->
  (∀ y : variable,
    y ∉ params -> var_free_in_expr y handler -> rename_variable r y = y) ->
  (∀ k' : continuation,
    cont_free_in_expr k' handler -> rename_continuation r k' = k') ->
  expr_eqb [] []
    (Let_cont_recursive
       (before ++ (k, Handler params metadata handler) :: after) body)
    (Let_cont_recursive
       (before
        ++ (k, rename_handler r (Handler params metadata handler)) :: after)
       body)
  = true.
Proof.
  intros Hclosed Hvars Hconts. simpl. rewrite !map_app. simpl.
  destruct (combine_names_refl (map fst before ++ k :: map fst after) []
              reflexive_map_nil)
    as (km' & -> & Hkm').
  rewrite (expr_eqb_refl body [] km' reflexive_map_nil Hkm'), andb_true_r.
  assert (Hsame :
    ∀ handlers : list (continuation * continuation_handler),
      forall2b (fun '(_, h) '(_, h') => handler_eqb [] km' h h') handlers
        handlers
      = true).
  { intros handlers. apply forall2b_refl. intros [k0 h0].
    apply handler_eqb_refl; [ apply reflexive_map_nil | exact Hkm' ]. }
  apply forall2b_app; [ apply Hsame | ].
  change
    (handler_eqb [] km' (Handler params metadata handler)
       (rename_handler r (Handler params metadata handler))
     && forall2b (fun '(_, h) '(_, h') => handler_eqb [] km' h h') after
          after
     = true).
  rewrite Hsame, andb_true_r. apply rename_handler_eqb; [ exact Hclosed | | ].
  - apply fixed_matched; [ apply reflexive_map_nil | ].
    intros x [Hnotin Hx]. exact (Hvars x Hnotin Hx).
  - apply (fixed_conts_matched r km' _ Hkm' Hconts).
Qed.

(* The binders of closed code: the code is renamed whole, and the other
   bindings of the group are left as they are. *)
Lemma rename_code_accepted (r : renaming) (before : list static_binding)
    (id : code_id) (c : code) (after : list static_binding) (body : expr) :
  code_is_closed c = true ->
  expr_eqb [] [] (Let_static (before ++ Bind_code id c :: after) body)
    (Let_static (before ++ Bind_code id (rename_code r c) :: after) body)
  = true.
Proof.
  intros Hclosed. simpl.
  rewrite expr_eqb_refl, andb_true_r by apply reflexive_map_nil.
  assert (Hsame :
    ∀ bindings : list static_binding,
      forall2b (binding_eqb []) bindings bindings = true)
    by (intros bindings; apply forall2b_refl; intros binding;
        apply binding_eqb_refl, reflexive_map_nil).
  apply forall2b_app; [ apply Hsame | ]. simpl.
  rewrite Pos.eqb_refl, (rename_code_eqb r c Hclosed), Hsame. reflexivity.
Qed.

(* ======================================================================== *)
(* The older forms of the statements about binders                          *)
(* ======================================================================== *)

(* These were statements of FlambdaNominalSpec.v: that the names a
   binder binds may be renamed. They are written with the relations
   [var_free_in_expr] and [cont_free_in_expr], which do not go into
   code. FlambdaCanonicalFreeNamesSpec.v has them with the free names
   of FlambdaBindersSpec.v, and those are the statements now; each is
   proved from the one here. These are no statement of a Spec file.
   They keep their names and their text because proofs use them. (The
   three statements about a binding that is not used are still
   statements of FlambdaNominalSpec.v.) *)

(* The variable of a [Let]: a renaming that fixes the free names of the
   body other than the bound variable may be applied to the bound variable
   and the body. The code that the body binds is closed, here and in the
   statements below. *)
Definition rename_let_statement : Prop :=
  ∀ (r : renaming) (x : variable) (kind : flambda_kind)
    (defining_expr : named) (body : expr),
    code_closed body = true ->
    (∀ y : variable,
      y <> x -> var_free_in_expr y body -> rename_variable r y = y) ->
    (∀ k : continuation,
      cont_free_in_expr k body -> rename_continuation r k = k) ->
    expr_equivalent (Let x kind defining_expr body)
      (Let (rename_variable r x) kind defining_expr (rename_expr r body)).

(* [Name_abstraction.pattern_match] on a [Let]: the bound variable becomes
   one that is not free in the body. *)
Definition freshen_let_statement : Prop :=
  ∀ (x fresh : variable) (kind : flambda_kind) (defining_expr : named)
    (body : expr),
    code_closed body = true ->
    ~ var_free_in_expr fresh body ->
    expr_equivalent (Let x kind defining_expr body)
      (Let fresh kind defining_expr
         (rename_expr (Renaming [(x, fresh)] []) body)).

(* The continuation of a non-recursive [Let_cont]. *)
Definition rename_let_cont_statement : Prop :=
  ∀ (r : renaming) (k : continuation) (handler : continuation_handler)
    (body : expr),
    code_closed body = true ->
    (∀ y : variable, var_free_in_expr y body -> rename_variable r y = y) ->
    (∀ k' : continuation,
      k' <> k -> cont_free_in_expr k' body -> rename_continuation r k' = k')
    ->
    expr_equivalent (Let_cont_non_recursive k handler body)
      (Let_cont_non_recursive (rename_continuation r k) handler
         (rename_expr r body)).

(* The parameters of a continuation handler. *)
Definition rename_handler_params_statement : Prop :=
  ∀ (r : renaming) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler body : expr),
    code_closed handler = true ->
    (∀ y : variable,
      y ∉ params -> var_free_in_expr y handler -> rename_variable r y = y) ->
    (∀ k' : continuation,
      cont_free_in_expr k' handler -> rename_continuation r k' = k') ->
    expr_equivalent
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k
         (rename_handler r (Handler params metadata handler)) body).

(* The parameters of one handler of a recursive [Let_cont]. *)
Definition rename_recursive_handler_params_statement : Prop :=
  ∀ (r : renaming) (before : list (continuation * continuation_handler))
    (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler : expr)
    (after : list (continuation * continuation_handler)) (body : expr),
    code_closed handler = true ->
    (∀ y : variable,
      y ∉ params -> var_free_in_expr y handler -> rename_variable r y = y) ->
    (∀ k' : continuation,
      cont_free_in_expr k' handler -> rename_continuation r k' = k') ->
    expr_equivalent
      (Let_cont_recursive
         (before ++ (k, Handler params metadata handler) :: after) body)
      (Let_cont_recursive
         (before
          ++ (k, rename_handler r (Handler params metadata handler)) :: after)
         body).

(* The continuations of a recursive [Let_cont], and the parameters of its
   handlers: a renaming that fixes the free names of the whole may be
   applied throughout. *)
Definition rename_let_cont_recursive_statement : Prop :=
  ∀ (r : renaming) (handlers : list (continuation * continuation_handler))
    (body : expr),
    code_closed (Let_cont_recursive handlers body) = true ->
    (∀ y : variable,
      var_free_in_expr y (Let_cont_recursive handlers body) ->
      rename_variable r y = y) ->
    (∀ k : continuation,
      cont_free_in_expr k (Let_cont_recursive handlers body) ->
      rename_continuation r k = k) ->
    expr_equivalent (Let_cont_recursive handlers body)
      (rename_expr r (Let_cont_recursive handlers body)).

Theorem rename_let : rename_let_statement.
Proof.
  intros r x kind d body Hclosed Hvars Hconts.
  apply expr_eqb_sound, rename_let_accepted; assumption.
Qed.

Theorem freshen_let : freshen_let_statement.
Proof.
  intros x fresh kind d body Hclosed Hfresh.
  pose proof (rename_let (Renaming [(x, fresh)] []) x kind d body Hclosed)
    as Hrename.
  replace (rename_variable (Renaming [(x, fresh)] []) x) with fresh
    in Hrename
    by (unfold rename_variable, swap; simpl; unfold swap;
        rewrite Pos.eqb_refl; reflexivity).
  apply Hrename.
  - intros y Hne Hy. unfold rename_variable. simpl. unfold swap.
    destruct (Pos.eqb_spec y x); [ contradiction | ].
    destruct (Pos.eqb_spec y fresh) as [-> | _]; [ contradiction | ].
    reflexivity.
  - intros k _. reflexivity.
Qed.

Theorem rename_let_cont : rename_let_cont_statement.
Proof.
  intros r k h body Hclosed Hvars Hconts.
  apply expr_eqb_sound, rename_let_cont_accepted; assumption.
Qed.

Theorem rename_handler_params : rename_handler_params_statement.
Proof.
  intros r k params metadata handler body Hclosed Hvars Hconts.
  apply expr_eqb_sound, rename_handler_params_accepted; assumption.
Qed.

Theorem rename_let_cont_recursive : rename_let_cont_recursive_statement.
Proof.
  intros r handlers body Hclosed Hvars Hconts.
  apply expr_eqb_sound, rename_expr_accepted; assumption.
Qed.

Theorem rename_recursive_handler_params :
  rename_recursive_handler_params_statement.
Proof.
  intros r before k params metadata handler after body Hclosed Hvars Hconts.
  apply expr_eqb_sound, rename_recursive_handler_params_accepted;
    assumption.
Qed.

Theorem rename_code_binders : rename_code_binders_statement.
Proof.
  intros r before id c after body Hclosed.
  apply expr_eqb_sound, rename_code_accepted, Hclosed.
Qed.

(* ======================================================================== *)
(* Bindings that are not used                                               *)
(* ======================================================================== *)

Theorem unused_let : unused_let_statement.
Proof.
  intros x kind d body Hheap Hunused vars conts statics hp rd.
  destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hd;
    [ destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind | ].
  - pose proof (exec_named_keeps_reads statics hp hp' rd rd' vars d v Hd)
      as ->.
    pose proof (exec_named_same_heap statics hp hp' rd rd vars d v Hheap Hd)
      as ->.
    apply equivalent_refines.
    apply
      (equivalent_trans _ (State body ((x, v) :: vars) conts statics hp rd));
      [ apply step_equivalent; rewrite step_let_exec, Hd, Hkind;
        reflexivity
      | ].
    apply coincide_equivalent.
    + intros y Hy. simpl.
      destruct (Pos.eqb_spec y x) as [-> | _]; [ contradiction | ].
      reflexivity.
    + intros k _. apply (conts_agree_same identity_renaming). reflexivity.
  - apply step_undefined_refines. rewrite step_let_exec, Hd, Hkind.
    reflexivity.
  - apply step_undefined_refines. rewrite step_let_exec, Hd. reflexivity.
Qed.

Theorem unused_let_cont : unused_let_cont_statement.
Proof.
  intros k h body Hunused vars conts statics hp rd.
  eapply equivalent_trans;
    [ apply step_equivalent, step_let_cont_non_recursive | ].
  apply coincide_equivalent; [ reflexivity | ].
  intros k' Hk'. unfold conts_agree_at, resolve.
  change (rename_continuation identity_renaming k') with k'. simpl.
  destruct (Pos.eqb_spec k' k) as [-> | _]; [ contradiction | ].
  exact (conts_agree_same identity_renaming k' conts eq_refl).
Qed.

Theorem unused_let_cont_recursive : unused_let_cont_recursive_statement.
Proof.
  intros handlers body Hstatics Hunused vars conts statics hp rd.
  eapply equivalent_trans;
    [ apply step_equivalent; rewrite step_let_cont_recursive, Hstatics;
      reflexivity
    | ].
  apply coincide_equivalent; [ reflexivity | ].
  intros k Hk. unfold conts_agree_at, resolve.
  change (rename_continuation identity_renaming k) with k.
  rewrite lookup_rec_closures_app, (Hunused k Hk).
  exact (conts_agree_same identity_renaming k conts eq_refl).
Qed.

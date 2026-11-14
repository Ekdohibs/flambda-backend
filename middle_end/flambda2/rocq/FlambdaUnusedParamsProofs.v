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

(* Proofs of the statements of FlambdaUnusedParamsSpec.v. Nothing here
   needs review.

   The pass changes a binder and every application in its scope, so it
   is not a rewrite of two expressions on their own. The term and its
   result are run side by side, by rounds
   (FlambdaRoundsProofs.rounds_refines): step for step, but for the
   [Let]s that the second has not got. The relation between the two
   states ([state_drops]) is:

   - the second expression is the first with parameters and arguments
     removed as some [masks] say, and with some [Let]s removed whose
     variables it does not use ([drops]). [drop_unused_params] removes
     no [Let]; FlambdaDeadParamsSpec.drop_dead_params, which is proved
     through the same relation (FlambdaDeadParamsProofs.v), does. This
     is a relation and not
     the function: where a function is called the two runs go into the
     same body, which is related to itself with every parameter kept
     ([drops_refl]), and the pass run on the term is one way of choosing
     the masks ([drop_params_drops]). There the mask of a recursive
     continuation is found from its handler as it stands; that the
     parameters it removes are not free in the handler once that is
     done is because the relation adds no free variable
     ([drops_free]);
   - the variables of the two states agree on the free variables of the
     second expression. They are not the same: the first run binds the
     parameters that the second has not got;
   - each continuation name stands, in the two states, for the same
     thing or for closures whose handlers are so related, whose
     variables so agree and whose continuations are in turn so related
     ([cont_drops]), the second taking only the parameters that the
     mask of the name keeps.

   Applying two such closures, the first to the values of all the
   arguments and the second to those kept, gives states so related
   ([bind_params_masked]): the parameters removed are not free in the
   second handler. The machine tests the kinds of the values against
   those that the handler gives (FlambdaSemanticsSpec.apply_cont_value);
   the second handler gives the kinds kept, so where the values are of
   the kinds of the first, those kept are of the kinds of the second
   ([kinds_masked]). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec FlambdaCompareSpec FlambdaCompareProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaRoundsProofs
  FlambdaUnusedParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Masks                                                                    *)
(* ======================================================================== *)

(* The elements at the positions where the mask says no. *)
Definition dropped {A : Type} (mask : list bool) (l : list A) : list A :=
  keep (map negb mask) l.

(* A mask is one for these parameters and this handler: it has an entry
   for each parameter, and the parameters it removes are not free in the
   handler. *)
Definition mask_fits (mask : option (list bool)) (params : list variable)
    (handler : expr) : Prop :=
  match mask with
  | None => True
  | Some mask =>
    length mask = length params
    ∧ ∀ x : variable, x ∈ dropped mask params -> ~ var_free_in_expr x handler
  end.

Lemma mask_of_head (known : masks) (k : continuation)
    (mask : option (list bool)) :
  mask_of ((k, mask) :: known) k = mask.
Proof. unfold mask_of. simpl. rewrite Pos.eqb_refl. reflexivity. Qed.

Lemma mask_of_other (known : masks) (k k' : continuation)
    (mask : option (list bool)) :
  k' <> k -> mask_of ((k, mask) :: known) k' = mask_of known k'.
Proof.
  intros Hne. unfold mask_of. simpl.
  destruct (Pos.eqb_spec k' k); [ contradiction | reflexivity ].
Qed.

(* The masks that the handlers of a recursive [Let_cont] add where all
   their parameters are kept. *)
Definition rec_masks (handlers : list (continuation * continuation_handler))
    : masks :=
  map (fun '(k, _) => (k, None)) handlers.

Lemma mask_of_rec_masks
    (handlers : list (continuation * continuation_handler)) (known : masks)
    (k : continuation) :
  mask_of (rec_masks handlers ++ known) k
  = match lookup k handlers with
    | Some _ => None
    | None => mask_of known k
    end.
Proof.
  unfold mask_of.
  induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
  destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

(* The masks that the handlers of a recursive [Let_cont] add come before
   those of the continuations in scope. *)
Lemma mask_of_app (loop known : masks) (k : continuation) :
  mask_of (loop ++ known) k
  = match lookup k loop with
    | Some mask => mask
    | None => mask_of known k
    end.
Proof.
  unfold mask_of.
  induction loop as [ | [k' mask] loop IH]; simpl; [ reflexivity | ].
  destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

(* An application whose continuation keeps all its parameters is not
   changed. *)
Lemma drop_args_none (known : masks) (ac : apply_cont) :
  mask_of known (ac_continuation ac) = None -> drop_args known ac = ac.
Proof.
  intros Hnone. unfold drop_args. rewrite Hnone. destruct ac. reflexivity.
Qed.

Lemma map_drop_args_none (known : masks) (arms : list (Z * apply_cont)) :
  (∀ k : continuation, mask_of known k = None) ->
  map (fun arm => (fst arm, drop_args known (snd arm))) arms = arms.
Proof.
  intros Hnone.
  induction arms as [ | [i ac] arms IH]; simpl; [ reflexivity | ].
  rewrite (drop_args_none known ac (Hnone _)), IH. reflexivity.
Qed.

Lemma find_arm_drop_args (known : masks) (i : Z)
    (arms : list (Z * apply_cont)) :
  find_arm i (map (fun arm => (fst arm, drop_args known (snd arm))) arms)
  = option_map (drop_args known) (find_arm i arms).
Proof.
  induction arms as [ | [discriminant ac] arms IH]; simpl; [ reflexivity | ].
  destruct (Z.eqb i discriminant); [ reflexivity | exact IH ].
Qed.

(* ======================================================================== *)
(* Evaluation under environments that agree                                 *)
(* ======================================================================== *)

Lemma eval_simple_agree (vars vars' : env) (s : simple) :
  (∀ x : variable, var_free_in_simple x s -> lookup x vars = lookup x vars') ->
  eval_simple vars' s = eval_simple vars s.
Proof.
  intros Hagree. destruct s as [x | sym | c]; simpl; try reflexivity.
  symmetry. apply Hagree. reflexivity.
Qed.

Lemma eval_simples_agree (vars vars' : env) (args : list simple) :
  (∀ x : variable, var_free_in_simples x args -> lookup x vars = lookup x vars')
  ->
  eval_simples vars' args = eval_simples vars args.
Proof.
  unfold var_free_in_simples.
  induction args as [ | s args IH]; intros Hagree; simpl; [ reflexivity | ].
  rewrite (eval_simple_agree vars vars' s), IH; [ reflexivity | | ].
  - intros x Hx. apply Hagree, Exists_cons_tl, Hx.
  - intros x Hx. apply Hagree, Exists_cons_hd, Hx.
Qed.

(* The arguments kept, under variables that agree on them, have the
   values kept. *)
Lemma eval_simples_keep (mask : list bool) :
  ∀ (args : list simple) (vars vars' : env) (vs : list value),
    eval_simples vars args = Some vs ->
    (∀ x : variable,
      var_free_in_simples x (keep mask args) -> lookup x vars = lookup x vars')
    ->
    eval_simples vars' (keep mask args) = Some (keep mask vs).
Proof.
  unfold var_free_in_simples.
  induction mask as [ | b mask IH]; intros args vars vars' vs Hargs Hagree;
    [ reflexivity | ].
  destruct args as [ | s args]; simpl in Hargs.
  - injection Hargs as <-. destruct b; reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ] eqn:Hrest;
      [ | discriminate ].
    injection Hargs as <-. destruct b; simpl in Hagree |- *.
    + rewrite (eval_simple_agree vars vars' s), Hs,
        (IH args vars vars' rest Hrest);
        [ reflexivity | | ].
      * intros x Hx. apply Hagree, Exists_cons_tl, Hx.
      * intros x Hx. apply Hagree, Exists_cons_hd, Hx.
    + exact (IH args vars vars' rest Hrest Hagree).
Qed.

Lemma eval_simples_masked (mask : option (list bool)) (args : list simple)
    (vars vars' : env) (vs : list value) :
  eval_simples vars args = Some vs ->
  (∀ x : variable,
    var_free_in_simples x (keep_all_or mask args) ->
    lookup x vars = lookup x vars') ->
  eval_simples vars' (keep_all_or mask args) = Some (keep_all_or mask vs).
Proof.
  destruct mask as [mask | ]; simpl; intros Hargs Hagree.
  - exact (eval_simples_keep mask args vars vars' vs Hargs Hagree).
  - rewrite (eval_simples_agree vars vars' args Hagree). exact Hargs.
Qed.

Lemma eval_named_agree (statics : static_env) (vars vars' : env) (d : named) :
  (∀ x : variable, var_free_in_named x d -> lookup x vars = lookup x vars') ->
  eval_named statics vars' d = eval_named statics vars d.
Proof.
  intros Hagree. symmetry.
  apply (eval_named_match [] vars vars' statics statics d d);
    [ | apply statics_match_same | apply named_eqb_refl, reflexive_map_nil ].
  intros x x' Hx Hmatch. apply names_match_nil in Hmatch. subst x'.
  apply Hagree, Hx.
Qed.

(* And it does the same on a heap. *)
Lemma exec_named_agree (statics : static_env) (hp : heap) (rd : read_record)
    (vars vars' : env)
    (d : named) :
  (∀ x : variable, var_free_in_named x d -> lookup x vars = lookup x vars') ->
  exec_named statics hp rd vars' d = exec_named statics hp rd vars d.
Proof.
  intros Hagree. apply exec_named_same.
  - exact (eval_named_agree statics vars vars' d Hagree).
  - intros p args ->. split; [ | reflexivity ].
    exact (eval_simples_agree vars vars' args Hagree).
Qed.

Lemma eval_static_bindings_agree (vars vars' : env)
    (bindings : list static_binding) :
  (∀ x : variable,
    Exists (var_free_in_static_binding x) bindings ->
    lookup x vars = lookup x vars') ->
  ∀ statics : static_env,
    eval_static_bindings vars' bindings statics
    = eval_static_bindings vars bindings statics.
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH];
    intros Hagree statics; simpl; [ reflexivity | | | ].
  - destruct (lookup id (static_code statics)); [ reflexivity | ].
    apply IH. intros x Hx. apply Hagree, Exists_cons_tl, Hx.
  - apply IH. intros x Hx. apply Hagree, Exists_cons_tl, Hx.
  - destruct (lookup s (static_values statics)); [ reflexivity | ].
    rewrite <- (eval_static_const_match [] vars vars' const const);
      [ | | apply static_const_eqb_refl, reflexive_map_nil ].
    + destruct (eval_static_const vars const); [ | reflexivity ].
      apply IH. intros x Hx. apply Hagree, Exists_cons_tl, Hx.
    + intros x x' Hx Hmatch. apply names_match_nil in Hmatch. subst x'.
      apply Hagree, Exists_cons_hd, Hx.
Qed.

(* ======================================================================== *)
(* Binding the parameters kept                                              *)
(* ======================================================================== *)

(* Binding the parameters kept to the values kept succeeds where binding
   them all does, and the two results agree on a variable that is not a
   parameter removed, if it is a parameter kept or the environments
   agreed on it. *)
Lemma bind_params_keep (params : list variable) :
  ∀ (mask : list bool) (vs : list value) (vars vars' bound : env),
    length mask = length params ->
    bind_params params vs vars = Some bound ->
    ∃ bound' : env,
      bind_params (keep mask params) (keep mask vs) vars' = Some bound'
      ∧ ∀ x : variable,
          x ∉ dropped mask params ->
          x ∈ keep mask params ∨ lookup x vars = lookup x vars' ->
          lookup x bound = lookup x bound'.
Proof.
  induction params as [ | p params IH];
    intros mask vs vars vars' bound Hlen Hbind.
  - destruct mask; [ | discriminate Hlen ].
    destruct vs; [ | discriminate Hbind ].
    injection Hbind as <-. exists vars'. split; [ reflexivity | ].
    intros x _ [[] | Hx]. exact Hx.
  - destruct mask as [ | b mask]; [ discriminate Hlen | ].
    destruct vs as [ | v vs]; [ discriminate Hbind | ].
    simpl in Hlen, Hbind. injection Hlen as Hlen.
    destruct b; unfold dropped; simpl.
    + destruct (IH mask vs ((p, v) :: vars) ((p, v) :: vars') bound Hlen Hbind)
        as (bound' & Hbind' & Hagree).
      exists bound'. split; [ exact Hbind' | ].
      intros x Hdropped Hx. apply Hagree; [ exact Hdropped | ]. simpl.
      destruct (Pos.eqb_spec x p) as [-> | Hne]; [ right; reflexivity | ].
      destruct Hx as [[Heq | Hin] | Hlookup];
        [ congruence | left; exact Hin | right; exact Hlookup ].
    + destruct (IH mask vs ((p, v) :: vars) vars' bound Hlen Hbind)
        as (bound' & Hbind' & Hagree).
      exists bound'. split; [ exact Hbind' | ].
      intros x Hdropped Hx. apply Hagree.
      * intros Hin. apply Hdropped. right. exact Hin.
      * simpl. destruct (Pos.eqb_spec x p) as [-> | Hne]; [ | exact Hx ].
        exfalso. apply Hdropped. left. reflexivity.
Qed.

(* The same for a mask that may keep everything, and a handler it fits:
   the two results agree on the free variables of the handler, if the
   environments agreed on those of the handler with its parameters. *)
Lemma bind_params_masked (mask : option (list bool))
    (params : list variable) (metadata' : handler_metadata) (handler' : expr)
    (vs : list value) (vars vars' bound : env) :
  mask_fits mask params handler' ->
  bind_params params vs vars = Some bound ->
  (∀ x : variable,
    var_free_in_handler x
      (Handler (keep_all_or mask params) metadata' handler') ->
    lookup x vars = lookup x vars') ->
  ∃ bound' : env,
    bind_params (keep_all_or mask params) (keep_all_or mask vs) vars'
    = Some bound'
    ∧ ∀ x : variable,
        var_free_in_expr x handler' -> lookup x bound = lookup x bound'.
Proof.
  intros Hfits Hbind Hvars. destruct mask as [mask | ]; simpl in *.
  - destruct Hfits as [Hlen Hdropped].
    destruct (bind_params_keep params mask vs vars vars' bound Hlen Hbind)
      as (bound' & Hbind' & Hagree).
    exists bound'. split; [ exact Hbind' | ].
    intros x Hx. apply Hagree.
    + intros Hin. exact (Hdropped x Hin Hx).
    + destruct (in_dec Pos.eq_dec x (keep mask params)) as [Hin | Hnotin];
        [ left; exact Hin | right; apply Hvars; split; assumption ].
  - destruct (bind_params_defined params vs vars vars' bound Hbind)
      as [bound' Hbind'].
    exists bound'. split; [ exact Hbind' | ].
    intros x Hx. symmetry.
    apply (bind_params_agree params vs vars vars' bound bound' Hbind Hbind').
    destruct (in_dec Pos.eq_dec x params) as [Hin | Hnotin];
      [ left; exact Hin | right; symmetry; apply Hvars; split; assumption ].
Qed.

(* The values kept are of the kinds kept. *)
Lemma map_keep {A B : Type} (g : A -> B) (mask : list bool) :
  ∀ l : list A, map g (keep mask l) = keep mask (map g l).
Proof.
  induction mask as [ | b mask IH]; intros [ | a l]; simpl; try reflexivity.
  - destruct b; reflexivity.
  - destruct b; simpl; rewrite IH; reflexivity.
Qed.

Lemma kinds_masked (mask : option (list bool)) (vs : list value)
    (kinds kinds' : list flambda_kind) :
  kinds' = keep_all_or mask kinds ->
  kinds_eqb (map kind_of_value vs) kinds = true ->
  kinds_eqb (map kind_of_value (keep_all_or mask vs)) kinds' = true.
Proof.
  intros -> Hkinds. apply kinds_eqb_eq in Hkinds. subst kinds.
  apply kinds_eqb_eq. destruct mask as [mask | ]; simpl;
    [ apply map_keep | reflexivity ].
Qed.

(* The kinds that [keep_kinds] leaves. *)
Lemma keep_kinds_kinds (mask : option (list bool))
    (metadata : handler_metadata) :
  hm_params_kinds (keep_kinds mask metadata)
  = keep_all_or mask (hm_params_kinds metadata).
Proof.
  destruct mask as [mask | ]; simpl; [ | reflexivity ].
  destruct (hm_is_exn_handler metadata); reflexivity.
Qed.

(* ======================================================================== *)
(* An expression with parameters and arguments removed                      *)
(* ======================================================================== *)

Definition handler_params (h : continuation_handler) : list variable :=
  match h with
  | Handler params _ _ => params
  end.

Definition handler_expr (h : continuation_handler) : expr :=
  match h with
  | Handler _ _ handler => handler
  end.

(* The second expression is the first with, at each [Let_cont], some
   parameters removed that the second handler does not use, their kinds
   removed with them, and their arguments removed where the
   continuation is applied; and with some [Let]s removed whose
   variables are not free in what follows them in the second and that
   are not of a primitive of the heap ([Drops_let_unused]). No [Apply]
   returns to a continuation that has lost a parameter. The handlers of
   a recursive [Let_cont] are related one with one, each with its mask
   ([handlers_drop]). (The warning silenced is about induction schemes,
   which nothing uses.) *)
#[local] Set Warnings "-register-all".
Inductive drops : masks -> expr -> expr -> Prop :=
  | Drops_let (known : masks) (x : variable) (kind : flambda_kind)
      (d : named) (body body' : expr) :
    drops known body body' ->
    drops known (Let x kind d body) (Let x kind d body')
  | Drops_let_static (known : masks) (bindings : list static_binding)
      (body body' : expr) :
    drops known body body' ->
    drops known (Let_static bindings body) (Let_static bindings body')
  | Drops_let_cont (known : masks) (k : continuation)
      (params : list variable) (metadata metadata' : handler_metadata)
      (handler handler' : expr) (mask : option (list bool))
      (body body' : expr) :
    drops known handler handler' ->
    mask_fits mask params handler' ->
    hm_params_kinds metadata' = keep_all_or mask (hm_params_kinds metadata) ->
    drops ((k, mask) :: known) body body' ->
    drops known
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k
         (Handler (keep_all_or mask params) metadata' handler') body')
  | Drops_let_cont_recursive (known loop : masks)
      (handlers handlers' : list (continuation * continuation_handler))
      (body body' : expr) :
    handlers_drop (loop ++ known) loop handlers handlers' ->
    handlers_bind_statics handlers' = handlers_bind_statics handlers ->
    drops (loop ++ known) body body' ->
    drops known (Let_cont_recursive handlers body)
      (Let_cont_recursive handlers' body')
  | Drops_apply (known : masks) (ap : apply_expr) :
    mask_of known (ap_continuation ap) = None ->
    mask_of known (ap_exn_continuation ap) = None ->
    drops known (Apply ap) (Apply ap)
  | Drops_apply_cont (known : masks) (ac : apply_cont) :
    drops known (Apply_cont ac) (Apply_cont (drop_args known ac))
  | Drops_switch (known : masks) (scrutinee : simple)
      (arms : list (Z * apply_cont)) :
    drops known (Switch scrutinee arms)
      (Switch scrutinee
         (map (fun arm => (fst arm, drop_args known (snd arm))) arms))
  | Drops_invalid (known : masks) : drops known Invalid Invalid
  | Drops_let_unused (known : masks) (x : variable) (kind : flambda_kind)
      (d : named) (body e' : expr) :
    named_changes_heap d = false ->
    drops known body e' ->
    ~ var_free_in_expr x e' ->
    drops known (Let x kind d body) e'
(* The handlers of a recursive [Let_cont], one with one, under the masks
   [inside] of where their expressions are: the same names, the masks
   [loop] of the names in their order, the second handler with the
   parameters that its mask keeps and their kinds, and expressions so
   related. *)
with handlers_drop :
    masks -> masks -> list (continuation * continuation_handler) ->
    list (continuation * continuation_handler) -> Prop :=
  | Handlers_drop_nil (inside : masks) : handlers_drop inside [] [] []
  | Handlers_drop_cons (inside : masks) (k : continuation)
      (mask : option (list bool)) (params : list variable)
      (metadata metadata' : handler_metadata) (handler handler' : expr)
      (loop : masks)
      (handlers handlers' : list (continuation * continuation_handler)) :
    drops inside handler handler' ->
    mask_fits mask params handler' ->
    hm_params_kinds metadata' = keep_all_or mask (hm_params_kinds metadata) ->
    handlers_drop inside loop handlers handlers' ->
    handlers_drop inside ((k, mask) :: loop)
      ((k, Handler params metadata handler) :: handlers)
      ((k, Handler (keep_all_or mask params) metadata' handler')
       :: handlers').

Lemma handlers_drop_lookup (inside loop : masks) (k : continuation)
    (handlers handlers' : list (continuation * continuation_handler)) :
  handlers_drop inside loop handlers handlers' ->
  match lookup k handlers, lookup k handlers', lookup k loop with
  | Some (Handler params metadata handler), Some h', Some mask =>
    ∃ (metadata' : handler_metadata) (handler' : expr),
      h' = Handler (keep_all_or mask params) metadata' handler'
      ∧ drops inside handler handler'
      ∧ mask_fits mask params handler'
      ∧ hm_params_kinds metadata'
        = keep_all_or mask (hm_params_kinds metadata)
  | None, None, None => True
  | _, _, _ => False
  end.
Proof.
  induction 1
    as [ inside
       | inside k1 mask params metadata metadata' handler handler' loop
           handlers handlers' Hhandler Hfits Hkinds _ IH ];
    simpl; [ exact I | ].
  destruct (Pos.eqb k k1); [ | exact IH ].
  exists metadata', handler'. auto.
Qed.

(* Every expression is related to itself, with every parameter kept. *)
Fixpoint drops_refl (e : expr) :
  ∀ known : masks,
    (∀ k : continuation, mask_of known k = None) -> drops known e e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros known Hnone.
  - apply Drops_let, drops_refl, Hnone.
  - apply Drops_let_static, drops_refl, Hnone.
  - apply (Drops_let_cont known k params metadata metadata handler handler
      None body body);
      [ apply drops_refl, Hnone | exact I | reflexivity | apply drops_refl ].
    intros k1. destruct (Pos.eq_dec k1 k) as [-> | Hne];
      [ apply mask_of_head | rewrite mask_of_other by exact Hne; apply Hnone ].
  - assert (Hnone' :
      ∀ k : continuation, mask_of (rec_masks handlers ++ known) k = None).
    { intros k. rewrite mask_of_rec_masks.
      destruct (lookup k handlers); [ reflexivity | apply Hnone ]. }
    apply (Drops_let_cont_recursive known (rec_masks handlers));
      [ | reflexivity | apply drops_refl, Hnone' ].
    revert Hnone'. generalize (rec_masks handlers ++ known).
    intros known' Hnone'.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      [ apply Handlers_drop_nil | ].
    apply (Handlers_drop_cons known' k None params metadata metadata handler
      handler);
      [ apply drops_refl, Hnone' | exact I | reflexivity | exact IH ].
  - apply Drops_apply; apply Hnone.
  - pose proof (Drops_apply_cont known ac) as Hdrops.
    rewrite (drop_args_none known ac (Hnone _)) in Hdrops. exact Hdrops.
  - pose proof (Drops_switch known scrutinee arms) as Hdrops.
    rewrite (map_drop_args_none known arms Hnone) in Hdrops. exact Hdrops.
  - apply Drops_invalid.
Qed.

(* ======================================================================== *)
(* The relation adds no free variable                                       *)
(* ======================================================================== *)

Lemma keep_in {A : Type} (mask : list bool) :
  ∀ (l : list A) (a : A), a ∈ keep mask l -> a ∈ l.
Proof.
  induction mask as [ | b mask IH]; intros l a; [ intros [] | ].
  destruct l as [ | a0 l]; [ destruct b; intros [] | ].
  destruct b; simpl.
  - intros [<- | Hin]; [ left; reflexivity | right; exact (IH l a Hin) ].
  - intros Hin. right. exact (IH l a Hin).
Qed.

Lemma in_keep_or_dropped {A : Type} (mask : list bool) :
  ∀ (l : list A) (a : A),
    length mask = length l -> a ∈ l -> a ∈ keep mask l ∨ a ∈ dropped mask l.
Proof.
  unfold dropped.
  induction mask as [ | b mask IH]; intros [ | a0 l] a Hlen Hin;
    try discriminate Hlen; [ destruct Hin | ].
  simpl in Hlen. injection Hlen as Hlen. destruct b; simpl.
  - destruct Hin as [<- | Hin]; [ left; left; reflexivity | ].
    destruct (IH l a Hlen Hin) as [Hkept | Hgone];
      [ left; right; exact Hkept | right; exact Hgone ].
  - destruct Hin as [<- | Hin]; [ right; left; reflexivity | ].
    destruct (IH l a Hlen Hin) as [Hkept | Hgone];
      [ left; exact Hkept | right; right; exact Hgone ].
Qed.

Lemma var_free_in_simples_keep (x : variable) (mask : option (list bool))
    (args : list simple) :
  var_free_in_simples x (keep_all_or mask args) -> var_free_in_simples x args.
Proof.
  destruct mask as [mask | ]; simpl; [ | auto ].
  unfold var_free_in_simples. intros Hx.
  apply Exists_exists in Hx as (s & Hin & Hs). apply Exists_exists.
  exists s. split; [ exact (keep_in mask args s Hin) | exact Hs ].
Qed.

(* A variable that is free in the second handler and is not one of the
   parameters kept is not one of the parameters. *)
Lemma mask_fits_not_param (mask : option (list bool)) (params : list variable)
    (handler' : expr) (x : variable) :
  mask_fits mask params handler' ->
  x ∉ keep_all_or mask params ->
  var_free_in_expr x handler' ->
  x ∉ params.
Proof.
  destruct mask as [mask | ]; simpl; [ | auto ].
  intros [Hlen Hdropped] Hnotin Hfree Hin.
  destruct (in_keep_or_dropped mask params x Hlen Hin) as [Hkept | Hgone];
    [ exact (Hnotin Hkept) | exact (Hdropped x Hgone Hfree) ].
Qed.

(* What the relation says of an expression of each form. *)
Lemma drops_let_inv (known : masks) (x : variable) (kind : flambda_kind)
    (d : named) (body e' : expr) :
  drops known (Let x kind d body) e' ->
  (∃ body' : expr, e' = Let x kind d body' ∧ drops known body body')
  ∨ (drops known body e' ∧ ~ var_free_in_expr x e').
Proof. intros Hdrops. inversion Hdrops; subst; eauto. Qed.

Lemma drops_let_static_inv (known : masks) (bindings : list static_binding)
    (body e' : expr) :
  drops known (Let_static bindings body) e' ->
  ∃ body' : expr, e' = Let_static bindings body' ∧ drops known body body'.
Proof. intros Hdrops. inversion Hdrops; subst. eauto. Qed.

Lemma drops_let_cont_inv (known : masks) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler body e' : expr) :
  drops known
    (Let_cont_non_recursive k (Handler params metadata handler) body) e' ->
  ∃ (metadata' : handler_metadata) (handler' : expr)
    (mask : option (list bool)) (body' : expr),
    e'
    = Let_cont_non_recursive k
        (Handler (keep_all_or mask params) metadata' handler') body'
    ∧ drops known handler handler'
    ∧ mask_fits mask params handler'
    ∧ drops ((k, mask) :: known) body body'.
Proof. intros Hdrops. inversion Hdrops; subst. eauto 8. Qed.

Lemma drops_let_cont_recursive_inv (known : masks)
    (handlers : list (continuation * continuation_handler)) (body e' : expr) :
  drops known (Let_cont_recursive handlers body) e' ->
  ∃ (loop : masks) (handlers' : list (continuation * continuation_handler))
    (body' : expr),
    e' = Let_cont_recursive handlers' body'
    ∧ handlers_drop (loop ++ known) loop handlers handlers'
    ∧ drops (loop ++ known) body body'.
Proof. intros Hdrops. inversion Hdrops; subst. eauto 8. Qed.

Lemma drops_apply_inv (known : masks) (ap : apply_expr) (e' : expr) :
  drops known (Apply ap) e' -> e' = Apply ap.
Proof. intros Hdrops. inversion Hdrops; subst. reflexivity. Qed.

Lemma drops_apply_cont_inv (known : masks) (ac : apply_cont) (e' : expr) :
  drops known (Apply_cont ac) e' -> e' = Apply_cont (drop_args known ac).
Proof. intros Hdrops. inversion Hdrops; subst. reflexivity. Qed.

Lemma drops_switch_inv (known : masks) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (e' : expr) :
  drops known (Switch scrutinee arms) e' ->
  e'
  = Switch scrutinee
      (map (fun arm => (fst arm, drop_args known (snd arm))) arms).
Proof. intros Hdrops. inversion Hdrops; subst. reflexivity. Qed.

Lemma drops_invalid_inv (known : masks) (e' : expr) :
  drops known Invalid e' -> e' = Invalid.
Proof. intros Hdrops. inversion Hdrops; subst. reflexivity. Qed.

Lemma handlers_drop_nil_inv (inside loop : masks)
    (handlers' : list (continuation * continuation_handler)) :
  handlers_drop inside loop [] handlers' -> handlers' = [].
Proof. intros Hdrop. inversion Hdrop; subst. reflexivity. Qed.

Lemma handlers_drop_cons_inv (inside loop : masks) (k : continuation)
    (params : list variable) (metadata : handler_metadata) (handler : expr)
    (handlers all' : list (continuation * continuation_handler)) :
  handlers_drop inside loop ((k, Handler params metadata handler) :: handlers)
    all' ->
  ∃ (mask : option (list bool)) (metadata' : handler_metadata)
    (handler' : expr) (rest : masks)
    (handlers' : list (continuation * continuation_handler)),
    loop = (k, mask) :: rest
    ∧ all'
      = (k, Handler (keep_all_or mask params) metadata' handler') :: handlers'
    ∧ drops inside handler handler'
    ∧ mask_fits mask params handler'
    ∧ handlers_drop inside rest handlers handlers'.
Proof. intros Hdrop. inversion Hdrop; subst. eauto 10. Qed.

(* A variable that is free in the second expression is free in the
   first: arguments go, and a parameter that goes is not free in the
   second handler. *)
Fixpoint drops_free (e : expr) :
  ∀ (known : masks) (e' : expr) (x : variable),
    drops known e e' -> var_free_in_expr x e' -> var_free_in_expr x e.
Proof.
  destruct e
    as [ y kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros known e' x Hdrops.
  - destruct (drops_let_inv _ _ _ _ _ _ Hdrops)
      as [(body' & -> & Hbody) | [Hbody Hunused]].
    + simpl. intros [Hd | [Hne Hfree]]; [ left; exact Hd | right ].
      split; [ exact Hne | exact (drops_free body known body' x Hbody Hfree) ].
    + simpl. intros Hfree. right.
      split; [ | exact (drops_free body known e' x Hbody Hfree) ].
      intros ->. exact (Hunused Hfree).
  - destruct (drops_let_static_inv _ _ _ _ Hdrops) as (body' & -> & Hbody).
    simpl. intros [Hbindings | Hfree]; [ left; exact Hbindings | right ].
    exact (drops_free body known body' x Hbody Hfree).
  - destruct (drops_let_cont_inv _ _ _ _ _ _ _ Hdrops)
      as (metadata' & handler' & mask & body' & -> & Hhandler & Hfits & Hbody).
    simpl. intros [[Hnotin Hfree] | Hfree];
      [ left | right; exact (drops_free body _ body' x Hbody Hfree) ].
    split;
      [ exact (mask_fits_not_param mask params handler' x Hfits Hnotin Hfree)
      | exact (drops_free handler known handler' x Hhandler Hfree) ].
  - destruct (drops_let_cont_recursive_inv _ _ _ _ Hdrops)
      as (loop & handlers' & body' & -> & Hhandlers & Hbody).
    simpl. intros [Hfree | Hfree];
      [ left | right; exact (drops_free body _ body' x Hbody Hfree) ].
    clear Hbody Hdrops. remember (loop ++ known) as inside eqn:Hinside.
    clear Hinside. revert loop handlers' Hhandlers Hfree.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      intros loop handlers' Hhandlers Hfree.
    + rewrite (handlers_drop_nil_inv _ _ _ Hhandlers) in Hfree. exact Hfree.
    + destruct (handlers_drop_cons_inv _ _ _ _ _ _ _ _ Hhandlers)
        as (mask & metadata' & handler' & rest & rest' & -> & -> & Hhandler
            & Hfits & Hrest).
      simpl in Hfree |- *.
      destruct Hfree as [[Hnotin Hfree] | Hfree];
        [ left | right; exact (IH rest rest' Hrest Hfree) ].
      split;
        [ exact (mask_fits_not_param mask params handler' x Hfits Hnotin Hfree)
        | exact (drops_free handler inside handler' x Hhandler Hfree) ].
  - rewrite (drops_apply_inv _ _ _ Hdrops). auto.
  - rewrite (drops_apply_cont_inv _ _ _ Hdrops). simpl.
    apply var_free_in_simples_keep.
  - rewrite (drops_switch_inv _ _ _ _ Hdrops). simpl.
    intros [Hscrutinee | Harms]; [ left; exact Hscrutinee | right ].
    apply Exists_exists in Harms as (arm' & Hin & Hfree).
    apply in_map_iff in Hin as (arm & <- & Hin). apply Exists_exists.
    exists arm. split; [ exact Hin | ].
    exact (var_free_in_simples_keep x _ _ Hfree).
  - rewrite (drops_invalid_inv _ _ Hdrops). auto.
Qed.

(* A mask that fits a handler fits it once parameters and arguments are
   removed from it. *)
Lemma mask_fits_drops (mask : option (list bool)) (params : list variable)
    (known : masks) (handler handler' : expr) :
  mask_fits mask params handler ->
  drops known handler handler' ->
  mask_fits mask params handler'.
Proof.
  destruct mask as [mask | ]; simpl; [ | auto ].
  intros [Hlen Hdropped] Hdrops. split; [ exact Hlen | ].
  intros x Hin Hfree.
  exact (Hdropped x Hin (drops_free handler known handler' x Hdrops Hfree)).
Qed.

(* ======================================================================== *)
(* The pass relates a term to its result                                    *)
(* ======================================================================== *)

(* No [Apply] of the expression returns to a continuation that the masks
   take a parameter from. *)
Definition safe (known : masks) (e : expr) : Prop :=
  ∀ (k : continuation) (mask : list bool),
    mask_of known k = Some mask -> returned_to k e = false.

Fixpoint drop_params_binds_statics (e : expr) :
  ∀ known : masks, binds_statics (drop_params known e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros known; simpl; try reflexivity.
  - apply drop_params_binds_statics.
  - rewrite drop_params_binds_statics. reflexivity.
  - rewrite !drop_params_binds_statics. reflexivity.
  - rewrite drop_params_binds_statics. f_equal.
    generalize (Let_cont_recursive handlers body). intros whole.
    generalize
      (map (fun '(k, h) => (k, used_params_in_loop k h whole)) handlers
       ++ known).
    intros known'.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite drop_params_binds_statics, IH. reflexivity.
Qed.

(* The parameters that [used_params] removes are not free in the
   handler. *)
Lemma dropped_unused (handler : expr) (params : list variable)
    (x : variable) :
  x ∈ dropped (map (fun param => var_free_b param handler) params) params ->
  var_free_b x handler = false.
Proof.
  unfold dropped.
  induction params as [ | p params IH]; simpl; [ intros [] | ].
  destruct (var_free_b p handler) eqn:Hp; simpl; [ exact IH | ].
  intros [<- | Hin]; [ exact Hp | exact (IH Hin) ].
Qed.

(* The mask with the first parameter of an exception handler kept. *)
Definition exn_bucket_kept (is_exn_handler : bool) (mask : list bool)
    : list bool :=
  match is_exn_handler, mask with
  | true, _ :: mask => true :: mask
  | _, _ => mask
  end.

Lemma exn_bucket_kept_length (is_exn_handler : bool) (mask : list bool) :
  length (exn_bucket_kept is_exn_handler mask) = length mask.
Proof. destruct is_exn_handler, mask; reflexivity. Qed.

(* Keeping it removes no more. *)
Lemma dropped_exn_bucket_kept {A : Type} (is_exn_handler : bool)
    (mask : list bool) (l : list A) (a : A) :
  a ∈ dropped (exn_bucket_kept is_exn_handler mask) l -> a ∈ dropped mask l.
Proof.
  destruct is_exn_handler; [ | exact (fun Hin => Hin) ].
  destruct mask as [ | b mask]; [ exact (fun Hin => Hin) | ].
  destruct l as [ | a' l]; [ destruct b; exact (fun Hin => Hin) | ].
  unfold dropped. simpl. destruct b; simpl; [ exact (fun Hin => Hin) | ].
  intros Hin. right. exact Hin.
Qed.

Lemma used_params_fit (is_exn_handler : bool) (params : list variable)
    (handler : expr) :
  mask_fits (used_params is_exn_handler params handler) params handler.
Proof.
  unfold used_params.
  change
    (match is_exn_handler, map (fun param => var_free_b param handler) params
     with
     | true, _ :: mask => true :: mask
     | _, _ => map (fun param => var_free_b param handler) params
     end)
    with
      (exn_bucket_kept is_exn_handler
         (map (fun param => var_free_b param handler) params)).
  destruct
    (forallb (fun used => used)
       (exn_bucket_kept is_exn_handler
          (map (fun param => var_free_b param handler) params)));
    [ exact I | ].
  split; [ rewrite exn_bucket_kept_length; apply length_map | ].
  intros x Hin Hfree. apply dropped_exn_bucket_kept, dropped_unused in Hin.
  apply var_free_b_sound in Hfree. congruence.
Qed.

(* The masks that the pass gives the handlers of a recursive [Let_cont]. *)
Lemma lookup_loop_masks (whole : expr) (k : continuation)
    (handlers : list (continuation * continuation_handler))
    (found : option (list bool)) :
  lookup k (map (fun '(k, h) => (k, used_params_in_loop k h whole)) handlers)
  = Some found ->
  ∃ h : continuation_handler, found = used_params_in_loop k h whole.
Proof.
  induction handlers as [ | [k' h] handlers IH]; simpl; [ discriminate | ].
  destruct (Pos.eqb_spec k k') as [-> | _];
    [ intros [= <-]; exists h; reflexivity | exact IH ].
Qed.

Lemma used_params_in_loop_some (k : continuation) (h : continuation_handler)
    (whole : expr) (mask : list bool) :
  used_params_in_loop k h whole = Some mask -> returned_to k whole = false.
Proof.
  destruct h as [params metadata handler]. simpl.
  destruct (returned_to k whole); [ discriminate | reflexivity ].
Qed.

Fixpoint drop_params_drops (e : expr) :
  ∀ known : masks, safe known e -> drops known e (drop_params known e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros known Hsafe; simpl.
  - apply Drops_let, drop_params_drops. exact Hsafe.
  - apply Drops_let_static, drop_params_drops. exact Hsafe.
  - apply Drops_let_cont.
    + apply drop_params_drops. intros k1 mask Hmask.
      specialize (Hsafe k1 mask Hmask). simpl in Hsafe.
      apply orb_false_iff in Hsafe as [Hhandler _]. exact Hhandler.
    + destruct (returned_to k body); [ exact I | apply used_params_fit ].
    + apply keep_kinds_kinds.
    + apply drop_params_drops. intros k1 mask Hmask.
      destruct (Pos.eq_dec k1 k) as [-> | Hne].
      * rewrite mask_of_head in Hmask.
        destruct (returned_to k body); [ discriminate Hmask | reflexivity ].
      * rewrite mask_of_other in Hmask by exact Hne.
        specialize (Hsafe k1 mask Hmask). simpl in Hsafe.
        apply orb_false_iff in Hsafe as [_ Hbody]. exact Hbody.
  - set (whole := Let_cont_recursive handlers body).
    set (loop :=
      map (fun '(k, h) => (k, used_params_in_loop k h whole)) handlers).
    assert (Hsafe' :
      ∀ (k : continuation) (mask : list bool),
        mask_of (loop ++ known) k = Some mask ->
        existsb (fun '(_, h) => returned_to_in_handler k h) handlers = false
        ∧ returned_to k body = false).
    { intros k mask. rewrite mask_of_app.
      destruct (lookup k loop) as [found | ] eqn:Hlookup.
      - intros ->. destruct (lookup_loop_masks whole k handlers _ Hlookup)
          as [h Hfound].
        symmetry in Hfound.
        apply used_params_in_loop_some in Hfound. simpl in Hfound.
        apply orb_false_iff in Hfound. exact Hfound.
      - intros Hmask. specialize (Hsafe k mask Hmask). simpl in Hsafe.
        apply orb_false_iff in Hsafe. exact Hsafe. }
    apply (Drops_let_cont_recursive known loop).
    + clear Hsafe. revert Hsafe'. generalize (loop ++ known). subst loop.
      generalize whole. clear whole. intros whole known' Hsafe'.
      induction handlers as [ | [k [params metadata handler]] handlers IH];
        simpl; [ apply Handlers_drop_nil | ].
      assert (Hhandler : drops known' handler (drop_params known' handler)).
      { apply drop_params_drops. intros k1 mask Hmask.
        destruct (Hsafe' k1 mask Hmask) as [Hhandlers _]. simpl in Hhandlers.
        apply orb_false_iff in Hhandlers as [Hhandler _]. exact Hhandler. }
      apply (Handlers_drop_cons known' k
        (used_params_in_loop k (Handler params metadata handler) whole) params
        metadata
        (keep_kinds
           (used_params_in_loop k (Handler params metadata handler) whole)
           metadata)
        handler (drop_params known' handler)).
      * exact Hhandler.
      * apply (mask_fits_drops _ _ known' handler); [ | exact Hhandler ].
        simpl.
        destruct (returned_to k whole); [ exact I | apply used_params_fit ].
      * apply keep_kinds_kinds.
      * apply IH. intros k1 mask Hmask.
        destruct (Hsafe' k1 mask Hmask) as [Hhandlers Hbody].
        simpl in Hhandlers.
        apply orb_false_iff in Hhandlers as [_ Hrest].
        exact (conj Hrest Hbody).
    + clear Hsafe Hsafe'. unfold handlers_bind_statics.
      generalize (loop ++ known). subst loop. generalize whole. clear whole.
      intros whole known'.
      induction handlers as [ | [k [params metadata handler]] handlers IH];
        simpl; [ reflexivity | ].
      rewrite drop_params_binds_statics, IH. reflexivity.
    + apply drop_params_drops. intros k mask Hmask.
      exact (proj2 (Hsafe' k mask Hmask)).
  - apply Drops_apply.
    + destruct (mask_of known (ap_continuation ap)) as [mask | ] eqn:Hmask;
        [ | reflexivity ].
      specialize (Hsafe _ mask Hmask). simpl in Hsafe.
      rewrite Pos.eqb_refl in Hsafe. discriminate Hsafe.
    + destruct (mask_of known (ap_exn_continuation ap)) as [mask | ]
        eqn:Hmask; [ | reflexivity ].
      specialize (Hsafe _ mask Hmask). simpl in Hsafe.
      rewrite Pos.eqb_refl, orb_true_r in Hsafe. discriminate Hsafe.
  - apply Drops_apply_cont.
  - apply Drops_switch.
  - apply Drops_invalid.
Qed.

(* ======================================================================== *)
(* Continuations and states                                                 *)
(* ======================================================================== *)

(* What a continuation name stands for in the two runs, given the mask
   of the name: the same thing, if all its parameters are kept; or
   closures whose handlers are related under the masks of where they
   were defined, the second with the parameters that the mask keeps and
   their kinds, under variables that agree on the free variables of the
   second
   handler and of the handlers defined with it, and under continuations
   that are in turn so related; or two that are related with all their
   parameters kept, with the same extra arguments; or what is left of
   the same over-application, to return and to raise to two that are
   related with all their parameters kept. *)
Inductive cont_drops : option (list bool) -> cont_value -> cont_value -> Prop :=
  | Same_drops (c : cont_value) : cont_drops None c c
  | Closure_drops (mask : option (list bool)) (known loop : masks)
      (params : list variable) (metadata metadata' : handler_metadata)
      (handler handler' : expr) (vars vars' : env) (conts conts' : cont_env)
      (recursive recursive' : list (continuation * continuation_handler)) :
    handlers_drop (loop ++ known) loop recursive recursive' ->
    drops (loop ++ known) handler handler' ->
    mask_fits mask params handler' ->
    hm_params_kinds metadata' = keep_all_or mask (hm_params_kinds metadata) ->
    (∀ x : variable,
      var_free_in_handler x
        (Handler (keep_all_or mask params) metadata' handler')
      ∨ any_handler (var_free_in_handler x) recursive' ->
      lookup x vars = lookup x vars') ->
    (∀ k : continuation,
      cont_drops (mask_of known k) (resolve k conts) (resolve k conts')) ->
    cont_drops mask
      (Closure (Handler params metadata handler) vars conts recursive)
      (Closure (Handler (keep_all_or mask params) metadata' handler') vars'
         conts' recursive')
  | Extra_args_drops (c c' : cont_value) (extra : list value) :
    cont_drops None c c' ->
    cont_drops None (Extra_args c extra) (Extra_args c' extra)
  | Over_application_drops (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    cont_drops None return_to return_to' ->
    cont_drops None exn_to exn_to' ->
    cont_drops None
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining remaining_arity return_arity return_to'
         exn_to').

Lemma with_extra_args_drops (c c' : cont_value) (extra : list value) :
  cont_drops None c c' ->
  cont_drops None (with_extra_args c extra) (with_extra_args c' extra).
Proof.
  intros Hc. destruct extra; simpl; [ exact Hc | apply Extra_args_drops, Hc ].
Qed.

Definition conts_drop (known : masks) (conts conts' : cont_env) : Prop :=
  ∀ k : continuation,
    cont_drops (mask_of known k) (resolve k conts) (resolve k conts').

Inductive state_drops : state -> state -> Prop :=
  | State_drops (known : masks) (e e' : expr) (vars vars' : env)
      (conts conts' : cont_env) (statics : static_env) (hp : heap)
        (rd : read_record) :
    drops known e e' ->
    (∀ x : variable, var_free_in_expr x e' -> lookup x vars = lookup x vars') ->
    conts_drop known conts conts' ->
    state_drops (State e vars conts statics hp rd)
      (State e' vars' conts' statics hp rd).

Lemma state_drops_refl (s : state) : state_drops s s.
Proof.
  destruct s as [e vars conts statics].
  apply (State_drops []);
    [ apply drops_refl; reflexivity | reflexivity
    | intros k; apply Same_drops ].
Qed.

(* The closures of the handlers of a recursive [Let_cont], over
   continuations that are related. *)
Lemma rec_conts_drop (known loop : masks)
    (handlers handlers' : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  handlers_drop (loop ++ known) loop handlers handlers' ->
  (∀ x : variable,
    any_handler (var_free_in_handler x) handlers' ->
    lookup x vars = lookup x vars') ->
  conts_drop known conts conts' ->
  conts_drop (loop ++ known)
    (rec_closures handlers vars conts ++ conts)
    (rec_closures handlers' vars' conts' ++ conts').
Proof.
  intros Hhandlers Hvars Hconts k.
  rewrite mask_of_app, !resolve_rec_closures.
  pose proof (handlers_drop_lookup _ _ k _ _ Hhandlers) as Hlookup.
  destruct (lookup k handlers) as [[params metadata handler] | ],
    (lookup k handlers') as [h' | ] eqn:Hl', (lookup k loop) as [mask | ];
    try contradiction; [ | apply Hconts ].
  destruct Hlookup as (metadata' & handler' & -> & Hhandler & Hfits & Hkinds).
  apply (Closure_drops mask known loop params metadata metadata' handler
    handler' vars vars' conts conts' handlers handlers');
    [ exact Hhandlers | exact Hhandler | exact Hfits | exact Hkinds |
    | exact Hconts ].
  intros x [Hx | Hx]; apply Hvars; [ | exact Hx ].
  exact (any_handler_in _ k _ _ (lookup_in _ _ _ Hl') Hx).
Qed.

(* The same full call, returning and raising to related continuations. *)
Lemma call_code_drops (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_drops None return_to return_to' ->
  cont_drops None exn_to exn_to' ->
  result_simulated state_drops
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure c vs args_arity return_arity return_to' exn_to'
       statics hp rd).
Proof.
  intros Hreturn Hraise.
  destruct (call_code_cases closure c vs args_arity return_arity statics hp rd)
    as [Hundefined | (callee_vars & Hcall)].
  - rewrite !Hundefined. exact I.
  - rewrite !Hcall. simpl.
    apply (State_drops []);
      [ apply drops_refl; reflexivity | reflexivity | ].
    intros k'.
    destruct (Pos.eq_dec k' (code_return_continuation c)) as [-> | Hne];
      [ rewrite !resolve_head; exact Hreturn | ].
    rewrite !(resolve_other k' (code_return_continuation c)) by exact Hne.
    destruct (Pos.eq_dec k' (code_exn_continuation c)) as [-> | Hne'];
      [ rewrite !resolve_head; exact Hraise | ].
    rewrite !(resolve_other k' (code_exn_continuation c)) by exact Hne'.
    apply Same_drops.
Qed.

(* The same call of any shape, where the two return continuations take
   a partial application to related results. *)
Lemma run_call_drops (shape : option call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_drops None return_to return_to' ->
  cont_drops None exn_to exn_to' ->
  (∀ partial : value,
    result_simulated state_drops
      (apply_cont_value return_to [partial] statics hp rd)
      (apply_cont_value return_to' [partial] statics hp rd)) ->
  result_simulated state_drops
    (run_call
       (fun partial => apply_cont_value return_to [partial] statics hp rd)
       shape return_arity return_to exn_to statics hp rd)
    (run_call
       (fun partial => apply_cont_value return_to' [partial] statics hp rd)
       shape return_arity return_to' exn_to' statics hp rd).
Proof.
  intros Hreturn Hraise Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl.
  - apply call_code_drops; assumption.
  - apply call_code_drops;
      [ apply Over_application_drops; assumption | exact Hraise ].
  - apply Hpartial.
  - exact I.
Qed.

(* Related continuations, applied: the second to the arguments that its
   mask keeps. *)
Lemma apply_cont_value_drops (mask : option (list bool))
    (c c' : cont_value) (statics : static_env) (hp : heap) (rd : read_record) :
  cont_drops mask c c' ->
  ∀ vs : list value,
    result_simulated state_drops (apply_cont_value c vs statics hp rd)
      (apply_cont_value c' (keep_all_or mask vs) statics hp rd).
Proof.
  intros Hc.
  induction Hc
    as [c
       | mask known' loop params metadata metadata' handler handler'
           def_vars def_vars' def_conts def_conts' recursive recursive'
           Hrecursive Hhandler Hfits Hkinds Hdef_vars Hdef_conts _
       | c c' extra Hc IHc
       | remaining remaining_arity return_arity return_to return_to' exn_to
           exn_to' Hreturn IHreturn Hraise _];
    intros vs.
  - (* the same continuation *)
    simpl.
    destruct (apply_cont_value c vs statics hp rd) as [s1 | k1 vs1 | ]; simpl;
      [ apply state_drops_refl | auto | exact I ].
  - (* related closures *)
    simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
      eqn:Hgiven; [ | exact I ].
    rewrite (kinds_masked mask vs _ _ Hkinds Hgiven).
    destruct (bind_params params vs def_vars) as [bound | ] eqn:Hbind;
      [ | exact I ].
    destruct (bind_params_masked mask params metadata' handler' vs def_vars
                def_vars' bound Hfits Hbind)
      as (bound' & Hbind' & Hagree);
      [ intros y Hy; apply Hdef_vars; left; exact Hy | ].
    rewrite Hbind'. simpl.
    apply (State_drops (loop ++ known'));
      [ exact Hhandler | exact Hagree | ].
    apply rec_conts_drop; [ exact Hrecursive | | exact Hdef_conts ].
    intros y Hy. apply Hdef_vars. right. exact Hy.
  - (* the same extra arguments *)
    exact (IHc (vs ++ extra)).
  - (* what is left of the same over-application *)
    cbn [keep_all_or]. rewrite !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; try exact I.
    unfold call_function.
    apply run_call_drops; [ exact Hreturn | exact Hraise | ].
    intros partial. exact (IHreturn [partial]).
Qed.

(* ======================================================================== *)
(* The simulation                                                           *)
(* ======================================================================== *)

(* Two related states step together, unless the first is at a [Let]
   that the second has not got. *)
Lemma drops_step (s s' : state) :
  state_drops s s' ->
  result_simulated state_drops (step s) (step s')
  ∨ ∃ (known : masks) (x : variable) (kind : flambda_kind) (d : named)
      (body e' : expr) (vars vars' : env) (conts conts' : cont_env)
      (statics : static_env) (hp : heap) (rd : read_record),
      s = State (Let x kind d body) vars conts statics hp rd
      ∧ s' = State e' vars' conts' statics hp rd
      ∧ named_changes_heap d = false
      ∧ drops known body e'
      ∧ ~ var_free_in_expr x e'
      ∧ (∀ y : variable,
          var_free_in_expr y e' -> lookup y vars = lookup y vars')
      ∧ conts_drop known conts conts'.
Proof.
  intros [known e e' vars vars' conts conts' statics hp rd Hdrops Hvars Hconts].
  destruct Hdrops
    as [ known x kind d body body' Hbody | known bindings body body' Hbody
       | known k params metadata metadata' handler handler' mask body body'
           Hhandler Hfits Hkinds Hbody
       | known loop handlers handlers' body body' Hhandlers Hbinds Hbody
       | known ap Hret Hexn | known ac | known scrutinee arms | known
       | known x kind d body e' Hheap Hbody Hunused ];
    [ left | left | left | left | left | left | left | left
    | right; do 13 eexists; repeat split; eassumption ];
    try rewrite !step_let_exec; autorewrite with step.
  - (* Let *)
    rewrite (exec_named_agree statics hp rd vars vars' d)
      by (intros y Hy; apply Hvars; left; exact Hy).
    destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ]; simpl;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind); simpl; [ | exact I ].
    apply (State_drops known); [ exact Hbody | | exact Hconts ].
    intros y Hy. simpl.
    destruct (Pos.eqb_spec y x) as [-> | Hne]; [ reflexivity | ].
    apply Hvars. right. split; assumption.
  - (* Let_static *)
    rewrite (eval_static_bindings_agree vars vars' bindings)
      by (intros y Hy; apply Hvars; left; exact Hy).
    destruct (eval_static_bindings vars bindings statics) as [extended | ];
      simpl; [ | exact I ].
    apply (State_drops known); [ exact Hbody | | exact Hconts ].
    intros y Hy. apply Hvars. right. exact Hy.
  - (* Let_cont_non_recursive *)
    simpl. apply (State_drops ((k, mask) :: known)); [ exact Hbody | | ].
    + intros y Hy. apply Hvars. right. exact Hy.
    + intros k'. destruct (Pos.eq_dec k' k) as [-> | Hne].
      * rewrite mask_of_head, !resolve_head.
        apply (Closure_drops mask known [] params metadata metadata' handler
          handler' vars vars' conts conts' [] []);
          [ apply Handlers_drop_nil | exact Hhandler | exact Hfits
          | exact Hkinds | | exact Hconts ].
        intros y [Hy | []]. apply Hvars. left. exact Hy.
      * rewrite mask_of_other, !resolve_other by exact Hne. apply Hconts.
  - (* Let_cont_recursive *)
    rewrite Hbinds.
    destruct (handlers_bind_statics handlers); simpl; [ exact I | ].
    apply (State_drops (loop ++ known)); [ exact Hbody | | ].
    + intros y Hy. apply Hvars. right. exact Hy.
    + apply rec_conts_drop; [ exact Hhandlers | | exact Hconts ].
      intros y Hy. apply Hvars. left. exact Hy.
  - (* Apply: the same call, returning to related continuations *)
    rewrite (eval_simple_agree vars vars' (ap_callee ap))
      by (intros y Hy; apply Hvars; left; exact Hy).
    rewrite (eval_simples_agree vars vars' (ap_args ap))
      by (intros y Hy; apply Hvars; right; left; exact Hy).
    rewrite (eval_simples_agree vars vars' (exn_extra_simples ap))
      by (intros y Hy; apply Hvars; right; right; exact Hy).
    destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | exact I ].
    pose proof (Hconts (ap_continuation ap)) as Hreturn.
    pose proof (Hconts (ap_exn_continuation ap)) as Hraise.
    rewrite Hret in Hreturn. rewrite Hexn in Hraise.
    unfold call_function.
    apply run_call_drops;
      [ exact Hreturn | apply with_extra_args_drops, Hraise | ].
    intros partial.
    exact (apply_cont_value_drops None _ _ statics _ rd_ Hreturn [partial]).
  - (* Apply_cont *)
    unfold drop_args. cbn [ac_args ac_continuation].
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs;
      [ | exact I ].
    rewrite (eval_simples_masked _ _ vars vars' vs Hargs)
      by (intros y Hy; apply Hvars; exact Hy).
    unfold apply_continuation.
    exact (apply_cont_value_drops _ _ _ statics _ rd_
             (Hconts (ac_continuation ac)) vs).
  - (* Switch *)
    rewrite (eval_simple_agree vars vars' scrutinee)
      by (intros y Hy; apply Hvars; left; exact Hy).
    destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    rewrite find_arm_drop_args.
    destruct (find_arm i arms) as [ac | ] eqn:Hfind; simpl; [ | exact I ].
    apply (State_drops known); [ apply Drops_apply_cont | | exact Hconts ].
    intros y Hy. apply Hvars. right. apply Exists_exists.
    exists (i, drop_args known ac). split; [ | exact Hy ].
    apply in_map_iff. exists (i, ac).
    split; [ reflexivity | apply find_arm_in, Hfind ].
  - (* Invalid *)
    exact I.
Qed.

(* A round: the [Let]s that the second state has not got, then one step
   of each. *)
Lemma drops_round_expr (e : expr) :
  ∀ (e' : expr) (vars vars' : env) (conts conts' : cont_env)
    (statics statics' : static_env) (hp : heap) (rd : read_record),
    state_drops (State e vars conts statics hp rd)
      (State e' vars' conts' statics' hp rd) ->
    ∃ n : nat,
      round_answered state_drops (run (S n) (State e vars conts statics hp rd))
        (State e' vars' conts' statics' hp rd).
Proof.
  induction e
    as [ x kind d body IH | bindings body _ | k h body _ | handlers body _
       | ap | ac | scrutinee arms | ];
    intros e' vars vars' conts conts' statics statics' hp rd Hrel;
    (destruct (drops_step _ _ Hrel)
       as [Hstep
          | (known & x0 & kind0 & d0 & body0 & e0 & vars0 & vars0' & conts0
             & conts0' & statics0 & hp0 & rd0 & Hs & Hs' & Hheap & Hbody
             & Hunused & Hvars & Hconts)];
     [ exists 0; apply step_round; exact Hstep | ]);
    try discriminate Hs.
  injection Hs as <- <- <- <- <- <- <- <-. injection Hs' as <- <- <- ->.
  destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hd.
  2: { exists 0. rewrite run_succ, step_let_exec, Hd. exact I. }
  pose proof (exec_named_keeps_reads statics hp hp' rd rd' vars d v Hd) as ->.
  pose proof (exec_named_same_heap statics hp hp' rd rd vars d v Hheap Hd)
    as ->.
  destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind.
  2: { exists 0. rewrite run_succ, step_let_exec, Hd, Hkind. exact I. }
  destruct (IH e' ((x, v) :: vars) vars' conts conts' statics statics hp rd)
    as [n Hn].
  { apply (State_drops known); [ exact Hbody | | exact Hconts ].
    intros y Hy. simpl.
    destruct (Pos.eqb_spec y x) as [-> | Hne];
      [ contradiction (Hunused Hy) | apply Hvars, Hy ]. }
  exists (S n). rewrite run_succ, step_let_exec, Hd, Hkind. exact Hn.
Qed.

Lemma drops_round (s s' : state) :
  state_drops s s' ->
  ∃ n : nat, round_answered state_drops (run (S n) s) s'.
Proof.
  intros Hrel.
  destruct Hrel
    as [known e e' vars vars' conts conts' statics hp Hdrops Hvars Hconts].
  apply drops_round_expr. apply (State_drops known); assumption.
Qed.

Theorem drop_unused_params_refines : drop_unused_params_refines_statement.
Proof.
  intros e vars conts statics hp rd.
  apply (rounds_refines state_drops drops_round).
  apply (State_drops []);
    [ apply drop_params_drops; intros k mask Hmask; discriminate Hmask
    | reflexivity | intros k; apply Same_drops ].
Qed.
